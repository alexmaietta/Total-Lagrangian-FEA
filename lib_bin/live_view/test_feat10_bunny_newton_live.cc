/**
 * FEAT10 Bunny Newton Live Demo
 *
 * Loads the FEAT10 bunny mesh of test_feat10_bunny_newton, clamps nodes near
 * the base, pulls the ears down for the first --load_steps steps, then
 * releases them, and shows the result live in Chrono Batch Render instead of
 * exporting VTU files. After every step, the surface nodes are copied from the
 * solver's device arrays into the renderer with one kernel (see
 * lib_src/live_view/LiveViewer.h). Press P to pause the solver.
 *
 * Example:
 *   bazel run //lib_bin/live_view:test_feat10_bunny_newton_live -- --res=2
 *   bazel run //lib_bin/live_view:test_feat10_bunny_newton_live -- \
 *     --steps=150 --screenshot=/tmp/bunny.png
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/FEAT10ConstraintManager.h"
#include "../../lib_src/solvers/SyncedNewton.cuh"
#include "../../lib_utils/cli_utils.h"
#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/quadrature_utils.h"
#include "../../lib_src/live_view/LiveViewer.h"

int main(int argc, char** argv) {
  ANCFCPUUtils::Cli cli(argv[0]);
  cli.SetDescription(
      "FEAT10 bunny Newton test shown live in Chrono Batch Render.");
  cli.AddInt("res", 0, "bunny mesh resolution (0, 2, 4, 8, 16, 26)");
  cli.AddInt("steps", 0, "steps to run (0 runs until the window is closed)");
  cli.AddInt("load_steps", 300, "steps the ears are pulled down for");
  cli.AddDouble("load", -100000.0, "total load on the ears (N)");
  LiveView::AddViewerOptions(cli);
  std::string cli_err;
  if (!cli.Parse(argc, argv, &cli_err) || cli.HelpRequested()) {
    if (!cli_err.empty()) {
      std::cerr << cli_err << "\n\n";
    }
    cli.PrintUsage(std::cout);
    return cli.HelpRequested() ? 0 : 1;
  }
  const int res        = cli.GetInt("res");
  const int steps      = cli.GetInt("steps");
  const int load_steps = cli.GetInt("load_steps");

  const std::string mesh_prefix =
      (res == 26) ? "data/meshes/T10/bunny_ascii_26.1"
                  : "data/meshes/T10/bunny_scaling/bunny_res" +
                        std::to_string(res) + ".1";
  Eigen::MatrixXd nodes;
  Eigen::MatrixXi elements;
  const int n_nodes =
      ANCFCPUUtils::FEAT10_read_nodes(mesh_prefix + ".node", nodes);
  const int n_elems =
      ANCFCPUUtils::FEAT10_read_elements(mesh_prefix + ".ele", elements);
  if (n_nodes <= 0 || n_elems <= 0) {
    std::cerr << "Failed to read mesh " << mesh_prefix << std::endl;
    return 1;
  }
  std::printf("mesh %s: %d nodes, %d elements\n", mesh_prefix.c_str(), n_nodes,
              n_elems);

  // Solver setup, as in test_feat10_bunny_newton
  GPU_FEAT10_Data data(n_elems, n_nodes);
  data.Initialize();

  Eigen::VectorXd x12 = nodes.col(0), y12 = nodes.col(1), z12 = nodes.col(2);

  std::vector<int> fixed;
  std::vector<int> loaded;
  for (int i = 0; i < n_nodes; ++i) {
    if (z12(i) < -0.4) {
      fixed.push_back(i);
    }
    if (z12(i) > 0.4) {
      loaded.push_back(i);
    }
  }
  FEAT10ConstraintManager constraints(&data);
  constraints.AddNodesToWorldCD(
      Eigen::Map<Eigen::VectorXi>(fixed.data(), Eigen::Index(fixed.size())));
  constraints.Finalize();

  Eigen::VectorXd f_ext = Eigen::VectorXd::Zero(data.get_n_coef() * 3);
  for (int i : loaded) {
    f_ext(3 * i + 2) = cli.GetDouble("load") / double(loaded.size());
  }
  data.SetExternalForce(f_ext);

  data.Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y, Quadrature::tet5pt_z,
             Quadrature::tet5pt_weights, x12, y12, z12, elements);
  data.ApplyMaterial(
      SolidMaterialProperties::SVK(3.0e8, 0.40, 920.0, 0.0, 0.0));
  data.CalcDnDuPre();
  data.CalcMassMatrix();
  data.CalcConstraintData();
  data.ConvertToCSR_ConstraintJacT();
  data.BuildConstraintJacobianCSR();
  data.CalcP();
  data.CalcInternalForce();

  SyncedNewtonParams params = {1e-3, 1e-4, 1e-4, 1e14, 5, 10, 1e-3, false};
  SyncedNewtonSolver solver(&data, data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);
  solver.AnalyzeHessianSparsity();
  solver.SetFixedSparsityPattern(true);

  // The viewer shows the bunny's boundary surface, following the solver nodes
  LiveView::LiveViewerOptions options =
      LiveView::ViewerOptions(cli, "TL-FEA bunny (Newton)");
  options.color_max = 0.3f;
  LiveView::LiveViewer viewer(options);
  viewer.AddSurface("Bunny", nodes, elements, {0, 0, n_nodes, n_elems, "bunny"},
                    data.GetX12DevicePtr(), data.GetY12DevicePtr(),
                    data.GetZ12DevicePtr());

  viewer.Run(steps, [&](int step) {
    if (step == load_steps) {
      data.SetExternalForce(Eigen::VectorXd::Zero(data.get_n_coef() * 3));
    }
    solver.Solve();
  });

  data.Destroy();
  return 0;
}
