/**
 * FEAT10 Teapot AdamW Live Demo
 *
 * Runs the FEAT10 teapot AdamW demo (test_feat10_teapot_adamw) and
 * shows it live in Chrono Batch Render instead of exporting VTU files.
 * Nodes in the bottom 20% of the mesh are fixed and an upward force is
 * applied to the top 10%. The original runs 6 steps; this one runs until
 * the window is closed by default. Press P to pause the solver.
 *
 * Example:
 *   bazel run //lib_bin/live_view:test_feat10_teapot_adamw_live
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/live_view/LiveViewer.h"
#include "../../lib_src/solvers/FEAT10ConstraintManager.h"
#include "../../lib_src/solvers/SyncedAdamWNocoop.cuh"
#include "../../lib_utils/cli_utils.h"
#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/quadrature_utils.h"

// Material properties (using SolidMaterialProperties)
const SolidMaterialProperties mat_teapot =
    SolidMaterialProperties::SVK(5.0e6,   // E: Young's modulus (Pa)
                                 0.35,    // nu: Poisson's ratio
                                 1000.0,  // rho0: Density (kg/m³)
                                 5.0e4,   // eta_damp (Pa·s)
                                 5.0e4    // lambda_damp (Pa·s)
    );

int main(int argc, char** argv) {
  ANCFCPUUtils::Cli cli(argv[0]);
  cli.SetDescription("FEAT10 teapot AdamW shown live in Chrono Batch Render.");
  cli.AddInt("steps", 0, "steps to run (0 runs until the window is closed)");
  LiveView::AddViewerOptions(cli);
  std::string cli_err;
  if (!cli.Parse(argc, argv, &cli_err) || cli.HelpRequested()) {
    if (!cli_err.empty()) {
      std::cerr << cli_err << "\n\n";
    }
    cli.PrintUsage(std::cout);
    return cli.HelpRequested() ? 0 : 1;
  }

  Eigen::MatrixXd nodes;
  Eigen::MatrixXi elements;

  const int n_nodes =
      ANCFCPUUtils::FEAT10_read_nodes("data/meshes/T10/teapot.1.node", nodes);
  const int n_elems = ANCFCPUUtils::FEAT10_read_elements(
      "data/meshes/T10/teapot.1.ele", elements);

  std::cout << "mesh read nodes: " << n_nodes << std::endl;
  std::cout << "mesh read elements: " << n_elems << std::endl;

  if (n_nodes <= 0 || n_elems <= 0) {
    std::cerr << "Failed to read teapot mesh." << std::endl;
    return 1;
  }

  GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);
  gpu_t10_data.Initialize();

  Eigen::VectorXd h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
  for (int i = 0; i < n_nodes; i++) {
    h_x12(i) = nodes(i, 0);
    h_y12(i) = nodes(i, 1);
    h_z12(i) = nodes(i, 2);
  }

  const double z_min = h_z12.minCoeff();
  const double z_max = h_z12.maxCoeff();
  const double z_rng = z_max - z_min;
  const double z_fix = z_min + 0.2 * z_rng;
  const double z_top = z_min + 0.9 * z_rng;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "z_min=" << z_min << " z_max=" << z_max << " z_fix<=" << z_fix
            << " z_top>=" << z_top << std::endl;

  // Fixed nodes: bottom 20% by height
  std::vector<int> fixed_node_indices;
  fixed_node_indices.reserve(static_cast<size_t>(n_nodes));
  for (int i = 0; i < h_z12.size(); ++i) {
    if (h_z12(i) <= z_fix) {
      fixed_node_indices.push_back(i);
    }
  }

  Eigen::VectorXi h_fixed_nodes(static_cast<int>(fixed_node_indices.size()));
  for (size_t i = 0; i < fixed_node_indices.size(); ++i) {
    h_fixed_nodes(static_cast<int>(i)) = fixed_node_indices[i];
  }
  std::cout << "Fixed nodes: " << h_fixed_nodes.size() << std::endl;
  FEAT10ConstraintManager constraint_manager(&gpu_t10_data);
  constraint_manager.AddNodesToWorldCD(h_fixed_nodes);
  constraint_manager.Finalize();

  // External force: total upward force distributed over top 10% by height
  Eigen::VectorXd h_f_ext(gpu_t10_data.get_n_coef() * 3);
  h_f_ext.setZero();

  std::vector<int> force_node_indices;
  force_node_indices.reserve(static_cast<size_t>(n_nodes));
  for (int i = 0; i < h_z12.size(); ++i) {
    if (h_z12(i) >= z_top) {
      force_node_indices.push_back(i);
    }
  }
  std::cout << "Force nodes: " << force_node_indices.size() << std::endl;

  constexpr double total_force_z = 50000.0;  // N
  if (!force_node_indices.empty()) {
    const double force_per_node = total_force_z / force_node_indices.size();
    for (int idx : force_node_indices) {
      h_f_ext(3 * idx + 2) = force_per_node;
    }
  }
  gpu_t10_data.SetExternalForce(h_f_ext);

  const Eigen::VectorXd& tet5pt_x_host       = Quadrature::tet5pt_x;
  const Eigen::VectorXd& tet5pt_y_host       = Quadrature::tet5pt_y;
  const Eigen::VectorXd& tet5pt_z_host       = Quadrature::tet5pt_z;
  const Eigen::VectorXd& tet5pt_weights_host = Quadrature::tet5pt_weights;

  gpu_t10_data.Setup(tet5pt_x_host, tet5pt_y_host, tet5pt_z_host,
                     tet5pt_weights_host, h_x12, h_y12, h_z12, elements);
  gpu_t10_data.ApplyMaterial(mat_teapot);

  gpu_t10_data.CalcDnDuPre();
  gpu_t10_data.CalcMassMatrix();
  gpu_t10_data.CalcConstraintData();
  gpu_t10_data.ConvertToCSR_ConstraintJacT();
  gpu_t10_data.CalcP();
  gpu_t10_data.CalcInternalForce();

  // More conservative AdamW settings for this deformation demo:
  // - Smaller lr + mild decay for stability.
  // - Enable relative inner convergence (inner_rtol) to avoid always hitting
  //   max_inner, and relax outer_tol since the constraint norm is an L2 over
  //   many fixed-node DOFs.
  // Quasi-static-style deformation: use a larger dt and a smaller constraint
  // penalty (rho) so AdamW can build up internal forces within each Solve().
  SyncedAdamWNocoopParams params = {2e-4, 0.9, 0.999, 1e-8, 0.0,  0.998, 1e-2,
                                    1e-4, 1e9, 1,     150,  5e-2, 10,    0.995};
  SyncedAdamWNocoopSolver solver(&gpu_t10_data,
                                 gpu_t10_data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);

  LiveView::LiveViewer viewer(
      LiveView::ViewerOptions(cli, "FEAT10 teapot AdamW"));
  viewer.AddSurface(
      "Teapot", nodes, elements, {0, 0, n_nodes, n_elems, "teapot"},
      gpu_t10_data.GetX12DevicePtr(), gpu_t10_data.GetY12DevicePtr(),
      gpu_t10_data.GetZ12DevicePtr());

  viewer.AddField({"Displacement", "m", LiveView::Displacement()});
  viewer.AddField({"von Mises stress", "Pa",
                   LiveView::ElementScalar(gpu_t10_data.GetVonMisesDevicePtr()),
                   0, 0, [&] { gpu_t10_data.ComputeVonMises(); }});

  viewer.Run(cli.GetInt("steps"), [&](int step) { solver.Solve(); });

  gpu_t10_data.Destroy();
  return 0;
}
