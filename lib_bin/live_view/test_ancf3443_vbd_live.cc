/**
 * ANCF3443 Shell VBD Live Demo
 *
 * Runs the ANCF3443 shell sag demo (beam_sag/test_ancf3443) with the VBD
 * solver and shows the shell live in Chrono Batch Render instead of
 * exporting VTU files. The surface is evaluated from the solver's nodal
 * coefficients through the element shape functions, so it bends with the
 * elements. The tip load is released after step 100, and the run lasts
 * --steps steps (1200 by default, where the original runs 200).
 *
 * The ground plane is hidden by default, since the deflection passes through
 * it; --ground shows it.
 *
 * Example:
 *   bazel run //lib_bin/live_view:test_ancf3443_vbd_live -- --res=0
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../../lib_src/elements/ANCF3443Data.cuh"
#include "../../lib_src/live_view/ANCFSurfaces.h"
#include "../../lib_src/live_view/LiveViewer.h"
#include "../../lib_src/solvers/SyncedVBD.cuh"
#include "../../lib_utils/cli_utils.h"
#include "../../lib_utils/quadrature_utils.h"
#include "../../lib_utils/visualization_utils.h"

namespace {

constexpr double kE    = 7e8;
constexpr double kNu   = 0.33;
constexpr double kRho0 = 2700;

constexpr double kH              = 0.1;  // Thickness
constexpr double kShellXSize     = 4.0;
constexpr double kShellYSize     = 2.0;
constexpr int kForceReleaseStep  = 100;
constexpr int kVtuEvery          = 20;
constexpr const char* kVtuPrefix = "ancf3443";

// Elements along each side of the shell
int ResolutionFromRes(int res) {
  switch (res) {
    case 0:
      return 10;
    case 2:
      return 20;
    case 4:
      return 50;
    case 8:
      return 100;
    case 16:
      return 150;
    case 32:
      return 200;
  }
  return -1;
}

}  // namespace

int main(int argc, char** argv) {
  ANCFCPUUtils::Cli cli(argv[0]);
  cli.SetDescription(
      "ANCF3443 shell sag test (VBD) shown live in Chrono Batch Render. "
      "Fields: Displacement, von Mises stress, Velocity.");
  cli.AddInt("res", 0,
             "0, 2, 4, 8, 16 or 32 (10, 20, 50, 100, 150 or 200 elements a "
             "side)");
  cli.AddInt("steps", 1200, "number of Solve() calls");
  cli.AddDouble("dt", 5e-4, "time step passed to the solver");
  cli.AddDouble("tip_force_z", -5000.0 * kH,
                "total vertical force on the free edge (N)");
  cli.AddDouble("lrratio", 0.5, "share of the load at the -y end of the edge");
  cli.AddDouble("omega", 1.8, "VBD relaxation factor");
  cli.AddOptionalString("csv", "",
                        "write the tip displacement CSV (to PATH if given)");
  cli.AddBool("vtu", false,
              "write VTU files to output/ancf3443/ every 20 steps");
  LiveView::AddViewerOptions(cli, /*show_ground=*/false);
  std::string cli_err;
  if (!cli.Parse(argc, argv, &cli_err) || cli.HelpRequested()) {
    if (!cli_err.empty()) {
      std::cerr << cli_err << "\n\n";
    }
    cli.PrintUsage(std::cout);
    return cli.HelpRequested() ? 0 : 1;
  }
  const int res                = cli.GetInt("res");
  const int steps              = cli.GetInt("steps");
  const double dt              = cli.GetDouble("dt");
  const double tip_force_z     = cli.GetDouble("tip_force_z");
  const double lrratio         = cli.GetDouble("lrratio");
  const double omega           = cli.GetDouble("omega");
  const bool write_csv         = cli.IsSet("csv");
  const bool write_vtu         = cli.GetBool("vtu");
  const int force_release_step = kForceReleaseStep;
  if (ResolutionFromRes(res) <= 0 || steps <= 0 || !(dt > 0.0) ||
      !(lrratio >= 0.0 && lrratio <= 1.0) || !(omega > 0.0)) {
    std::cerr << "Invalid option (see --help)\n";
    return 1;
  }

  const int derived_res = ResolutionFromRes(res);
  const int x_res       = derived_res;
  const int y_res       = derived_res;

  const std::string vtu_out_dir = "output/ancf3443";
  if (write_vtu) {
    std::filesystem::create_directories(vtu_out_dir);
  }

  const int n_elem  = x_res * y_res;
  const int n_nodes = (x_res + 1) * (y_res + 1);
  GPU_ANCF3443_Data data(n_nodes, n_elem);
  data.Initialize();

  const double L = kShellXSize / static_cast<double>(x_res);
  const double W = kShellYSize / static_cast<double>(y_res);
  const double H = kH;

  Eigen::VectorXd h_x12(data.get_n_coef());
  Eigen::VectorXd h_y12(data.get_n_coef());
  Eigen::VectorXd h_z12(data.get_n_coef());
  Eigen::MatrixXi element_connectivity(n_elem, 4);
  ANCFCPUUtils::ANCF3443_generate_shell_coordinates(
      kShellXSize, kShellYSize, x_res, y_res, h_x12, h_y12, h_z12,
      element_connectivity);

  const int n_nodes_from_coords = static_cast<int>(h_x12.size()) / 4;
  if (n_nodes_from_coords != n_nodes) {
    std::cerr << "Shell coordinate generator produced n_nodes="
              << n_nodes_from_coords << " but expected " << n_nodes << "\n";
    return 1;
  }
  std::vector<int> fixed_dofs;
  fixed_dofs.reserve(static_cast<size_t>(n_nodes));
  const double x_edge_tol = 1e-10 * std::max(1.0, std::abs(kShellXSize));
  for (int node = 0; node < n_nodes; ++node) {
    const int base = node * 4;
    if (std::abs(h_x12(base + 0) - 0.0) <= x_edge_tol) {
      for (int d = 0; d < 4; ++d) {
        fixed_dofs.push_back(base + d);
      }
    }
  }
  Eigen::VectorXi h_fixed_nodes(static_cast<int>(fixed_dofs.size()));
  for (int i = 0; i < static_cast<int>(fixed_dofs.size()); ++i) {
    h_fixed_nodes(i) = fixed_dofs[static_cast<size_t>(i)];
  }
  const int constrained_dofs = h_fixed_nodes.size() * 3;

  std::cout << "ANCF3443(shell): res=" << res << " nx=" << x_res
            << " ny=" << y_res << " elements=" << n_elem << " nodes=" << n_nodes
            << " coef=" << data.get_n_coef()
            << " constrained_dofs=" << constrained_dofs << " steps=" << steps
            << " dt=" << dt << " L=" << L << " W=" << W << " H=" << H
            << " x_size=" << kShellXSize << " y_size=" << kShellYSize
            << " tip_force_z=" << tip_force_z
            << " force_release_step=" << force_release_step << std::endl;

  data.SetNodalFixed(h_fixed_nodes);

  Eigen::VectorXd h_f_ext_on(data.get_n_coef() * 3);
  h_f_ext_on.setZero();
  std::vector<int> tip_nodes;
  tip_nodes.reserve(static_cast<size_t>(n_nodes));
  double tip_y_min = std::numeric_limits<double>::infinity();
  double tip_y_max = -std::numeric_limits<double>::infinity();
  for (int node = 0; node < n_nodes; ++node) {
    const int base = node * 4;
    if (std::abs(h_x12(base + 0) - kShellXSize) <= x_edge_tol) {
      tip_nodes.push_back(node);
      tip_y_min = std::min(tip_y_min, h_y12(base + 0));
      tip_y_max = std::max(tip_y_max, h_y12(base + 0));
    }
  }
  if (tip_nodes.empty()) {
    std::cerr << "No tip nodes found at x == x_size. Check mesh generation.\n";
    return 1;
  }

  // Distribute total vertical force across all tip-edge nodes. `lrratio`
  // provides a linear bias from -y (ratio) to +y (1-ratio). When lrratio=0.5,
  // this is uniform.
  const double denom_y =
      (tip_y_max > tip_y_min) ? (tip_y_max - tip_y_min) : 1.0;
  std::vector<double> weights(tip_nodes.size(), 0.0);
  double wsum = 0.0;
  for (size_t i = 0; i < tip_nodes.size(); ++i) {
    const int node = tip_nodes[i];
    const int base = node * 4;
    const double t = (h_y12(base + 0) - tip_y_min) / denom_y;  // in [0,1]
    const double w = (1.0 - t) * lrratio + t * (1.0 - lrratio);
    weights[i]     = w;
    wsum += w;
  }
  if (wsum <= 0.0) {
    std::cerr << "Invalid tip-edge load weights (sum <= 0).\n";
    return 1;
  }
  for (size_t i = 0; i < tip_nodes.size(); ++i) {
    const int node  = tip_nodes[i];
    const double fz = tip_force_z * (weights[i] / wsum);
    const int coeff = node * 4;  // position coefficient
    h_f_ext_on(coeff * 3 + 2) += fz;
  }
  Eigen::VectorXd h_f_ext_off(data.get_n_coef() * 3);
  h_f_ext_off.setZero();
  data.SetExternalForce(h_f_ext_on);

  auto maybe_release_force = [&](int step) {
    if (force_release_step >= 0 && step == force_release_step) {
      data.SetExternalForce(h_f_ext_off);
      std::cout << "Released tip force at step " << step << std::endl;
    }
  };

  data.Setup(L, W, H, Quadrature::gauss_xi_m_7, Quadrature::gauss_eta_m_7,
             Quadrature::gauss_zeta_m_3, Quadrature::gauss_xi_4,
             Quadrature::gauss_eta_4, Quadrature::gauss_zeta_3,
             Quadrature::weight_xi_m_7, Quadrature::weight_eta_m_7,
             Quadrature::weight_zeta_m_3, Quadrature::weight_xi_4,
             Quadrature::weight_eta_4, Quadrature::weight_zeta_3, h_x12, h_y12,
             h_z12, element_connectivity);

  data.SetDensity(kRho0);
  data.SetDamping(0.0, 0.0);
  data.SetSVK(kE, kNu);

  data.CalcDsDuPre();
  data.CalcMassMatrix();
  data.CalcConstraintData();
  data.ConvertToCSR_ConstraintJacT();
  data.BuildConstraintJacobianCSR();
  data.CalcP();
  data.CalcInternalForce();

  auto want_vtu = [&](int step) {
    return write_vtu && (step % kVtuEvery) == 0;
  };

  auto export_vtu = [&](int step, const Eigen::VectorXd& x12,
                        const Eigen::VectorXd& y12,
                        const Eigen::VectorXd& z12) {
    std::ostringstream oss;
    oss << vtu_out_dir << "/" << kVtuPrefix << "_vbd_" << std::setw(6)
        << std::setfill('0') << step << ".vtu";
    data.ComputeVonMises();
    Eigen::VectorXd vm;
    data.RetrieveVonMisesToCPU(vm);
    ANCFCPUUtils::VisualizationUtils::ExportANCF3443ToVTU(
        x12, y12, z12, element_connectivity, H, oss.str(), &vm);
  };

  if (want_vtu(0)) {
    Eigen::VectorXd x12, y12, z12;
    data.RetrievePositionToCPU(x12, y12, z12);
    export_vtu(0, x12, y12, z12);
  }

  std::vector<double> tip_z_history;
  if (write_csv) {
    tip_z_history.reserve(static_cast<size_t>(steps));
  }

  auto tip_z_average = [&](const Eigen::VectorXd& z12) {
    double sum = 0.0;
    for (int node : tip_nodes) {
      sum += z12(node * 4 + 0);
    }
    return sum / static_cast<double>(tip_nodes.size());
  };

  LiveView::LiveViewer viewer(
      LiveView::ViewerOptions(cli, "ANCF3443 shell (vbd)"));
  viewer.AddSurface("Shell",
                    LiveView::ANCF3443Surface(h_x12, h_y12, h_z12,
                                              element_connectivity, L, W, H),
                    data.GetX12DevicePtr(), data.GetY12DevicePtr(),
                    data.GetZ12DevicePtr());

  viewer.AddField({"Displacement", "m", LiveView::Displacement()});
  viewer.AddField({"von Mises stress", "Pa",
                   LiveView::ElementScalar(data.GetVonMisesDevicePtr()), 0, 0,
                   [&] { data.ComputeVonMises(); }});

  SyncedVBDParams params = {1e-4, 1e-4,  1e-4,  1e14, 5, 500,
                            dt,   omega, 1e-12, 25,   1};
  SyncedVBDSolver solver(&data, data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);
  solver.InitializeColoring();
  solver.InitializeMassDiagBlocks();
  solver.InitializeFixedMap();
  viewer.AddField({"Velocity", "m/s",
                   LiveView::NodeVector(solver.GetVelocityGuessDevicePtr())});

  viewer.Run(steps, [&](int step) {
    maybe_release_force(step);
    solver.Solve();
    const int out_step = step + 1;
    const bool do_vtu  = want_vtu(out_step);
    if (write_csv || do_vtu) {
      Eigen::VectorXd x12, y12, z12;
      data.RetrievePositionToCPU(x12, y12, z12);
      if (write_csv) {
        tip_z_history.push_back(tip_z_average(z12));
      }
      if (do_vtu) {
        export_vtu(out_step, x12, y12, z12);
      }
    }
  });

  if (write_csv) {
    std::string out_path = cli.GetString("csv");
    if (out_path.empty()) {
      out_path = "tip_z_history_ancf3443_vbd.csv";
    }
    std::ofstream csv(out_path);
    csv << std::fixed << std::setprecision(17) << "step,tip_z\n";
    for (size_t i = 0; i < tip_z_history.size(); ++i) {
      csv << i << "," << tip_z_history[i] << "\n";
    }
    std::cout << "Wrote " << out_path << std::endl;
  }

  data.Destroy();
  return 0;
}
