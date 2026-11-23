/**
 * ANCF3243 Beam VBD Live Demo
 *
 * Runs the ANCF3243 beam sag demo (beam_sag/test_ancf3243) with the VBD
 * solver and shows the beam live in Chrono Batch Render instead of
 * exporting VTU files. The surface is evaluated from the solver's nodal
 * coefficients through the element shape functions, so it bends with the
 * elements. The tip load is released at --force_release_step.
 *
 * The ground plane is hidden by default, since the deflection passes through
 * it; --ground shows it.
 *
 * Example:
 *   bazel run //lib_bin/live_view:test_ancf3243_vbd_live -- --res=2
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../lib_src/elements/ANCF3243Data.cuh"
#include "../../lib_src/live_view/ANCFSurfaces.h"
#include "../../lib_src/live_view/LiveViewer.h"
#include "../../lib_src/solvers/SyncedVBD.cuh"
#include "../../lib_utils/cli_utils.h"
#include "../../lib_utils/mesh_utils.h"
#include "../../lib_utils/quadrature_utils.h"
#include "../../lib_utils/visualization_utils.h"

namespace {

constexpr double kE    = 7e8;
constexpr double kNu   = 0.33;
constexpr double kRho0 = 2700;

constexpr double kL = 0.2;  // Element length
constexpr double kW = 0.1;  // Cross-section width
constexpr double kH = 0.1;  // Cross-section height

constexpr int kVtuEvery          = 20;
constexpr const char* kVtuPrefix = "ancf3243";

int NumElementsForRes(int res) {
  switch (res) {
    case 0:
      return 1000;
    case 2:
      return 10000;
    case 4:
      return 50000;
    case 8:
      return 100000;
    case 16:
      return 200000;
    case 32:
      return 500000;
  }
  return -1;
}

int SelectTipNodeByMaxX(const Eigen::VectorXd& x12, int n_nodes) {
  if (n_nodes <= 0) {
    return 0;
  }
  int tip_node = 0;
  double max_x = x12(0);
  for (int node = 1; node < n_nodes; ++node) {
    const double x = x12(4 * node);
    if (x > max_x) {
      max_x    = x;
      tip_node = node;
    }
  }
  return tip_node;
}

bool ValidateBeamConnectivityChain(const Eigen::MatrixXi& conn, int n_nodes,
                                   std::string* error) {
  const int n_elements = static_cast<int>(conn.rows());
  if (conn.cols() != 2) {
    if (error) {
      *error = "connectivity must have 2 columns";
    }
    return false;
  }
  if (n_elements <= 0) {
    if (error) {
      *error = "n_elements must be > 0";
    }
    return false;
  }
  if (n_nodes != n_elements + 1) {
    if (error) {
      std::ostringstream oss;
      oss << "expected n_nodes == n_elements + 1 for a 1D chain (n_nodes="
          << n_nodes << ", n_elements=" << n_elements << ")";
      *error = oss.str();
    }
    return false;
  }
  for (int e = 0; e < n_elements; ++e) {
    const int n0 = conn(e, 0);
    const int n1 = conn(e, 1);
    if (n0 != e || n1 != e + 1) {
      if (error) {
        std::ostringstream oss;
        oss << "unexpected chain connectivity at element " << e << ": (" << n0
            << "," << n1 << "), expected (" << e << "," << (e + 1) << ")";
        *error = oss.str();
      }
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  ANCFCPUUtils::Cli cli(argv[0]);
  cli.SetDescription(
      "ANCF3243 beam sag test (VBD) shown live in Chrono Batch Render. "
      "Fields: Displacement, von Mises stress, Velocity.");
  cli.AddInt("res", 0,
             "0, 2, 4, 8, 16 or 32 (1000, 10000, 50000, 100000, 200000 or "
             "500000 elements)");
  cli.AddInt("steps", 200, "number of Solve() calls");
  cli.AddDouble("dt", 1e-3, "time step passed to the solver");
  cli.AddDouble("tip_force_z", 5000.0, "vertical point force at the tip (N)");
  cli.AddInt("force_release_step", 100,
             "step the tip force is released at (-1 never)");
  cli.AddDouble("omega", 1.0, "VBD relaxation factor");
  cli.AddOptionalString("csv", "",
                        "write the tip displacement CSV (to PATH if given)");
  cli.AddBool("vtu", false,
              "write VTU files to output/ancf3243/ every 20 steps");
  cli.AddDouble("section_scale", 0.0,
                "draw the cross-section this many times its size (0 draws "
                "it at 2% of the beam's length)");
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
  const int num_elements       = NumElementsForRes(res);
  const int steps              = cli.GetInt("steps");
  const double dt              = cli.GetDouble("dt");
  const double tip_force_z     = cli.GetDouble("tip_force_z");
  const int force_release_step = cli.GetInt("force_release_step");
  const double omega           = cli.GetDouble("omega");
  const bool write_csv         = cli.IsSet("csv");
  const bool write_vtu         = cli.GetBool("vtu");
  if (num_elements <= 0 || steps <= 0 || !(dt > 0.0) ||
      force_release_step < -1 || !(omega > 0.0) ||
      cli.GetDouble("section_scale") < 0.0) {
    std::cerr << "Invalid option (see --help)\n";
    return 1;
  }

  const double L           = kL;
  const double W           = kW;
  const double H           = kH;
  const double beam_length = static_cast<double>(num_elements) * L;

  const std::string vtu_out_dir = "output/ancf3243";
  if (write_vtu) {
    std::filesystem::create_directories(vtu_out_dir);
  }

  ANCFCPUUtils::GridMeshGenerator grid_gen(beam_length, 0.0, L, true, false);
  grid_gen.generate_mesh();

  const int n_nodes    = grid_gen.get_num_nodes();
  const int n_elements = grid_gen.get_num_elements();

  Eigen::VectorXi h_fixed_nodes(4);
  h_fixed_nodes << 0, 1, 2, 3;
  const int constrained_dofs = h_fixed_nodes.size() * 3;

  std::cout << "ANCF3243: nodes=" << n_nodes << " elements=" << n_elements
            << " coef=" << (4 * n_nodes)
            << " constrained_dofs=" << constrained_dofs << " res=" << res
            << " num_elements=" << num_elements << " steps=" << steps
            << " dt=" << dt << " L=" << L << " W=" << W << " H=" << H
            << " tip_force_z=" << tip_force_z
            << " force_release_step=" << force_release_step << std::endl;

  GPU_ANCF3243_Data data(n_nodes, n_elements);
  data.Initialize();

  Eigen::VectorXd h_x12(data.get_n_coef());
  Eigen::VectorXd h_y12(data.get_n_coef());
  Eigen::VectorXd h_z12(data.get_n_coef());
  grid_gen.get_coordinates(h_x12, h_y12, h_z12);

  Eigen::MatrixXi h_element_connectivity;
  grid_gen.get_element_connectivity(h_element_connectivity);

  {
    std::string err;
    if (!ValidateBeamConnectivityChain(h_element_connectivity, n_nodes, &err)) {
      std::cerr << "GridMeshGenerator connectivity validation failed: " << err
                << std::endl;
      return 1;
    }
  }

  data.SetNodalFixed(h_fixed_nodes);

  Eigen::VectorXd h_f_ext_on(data.get_n_coef() * 3);
  h_f_ext_on.setZero();
  const int tip_node = SelectTipNodeByMaxX(h_x12, n_nodes);
  const int tip_coef = tip_node * 4;  // coef-slot 0 (position) at tip node
  const double tip_x = h_x12(tip_coef);
  std::cout << "Tip: node=" << tip_node << " coef=" << tip_coef
            << " x=" << tip_x << " (expected ~" << beam_length << ")"
            << std::endl;
  h_f_ext_on(tip_coef * 3 + 2) = tip_force_z;

  Eigen::VectorXd h_f_ext_off(data.get_n_coef() * 3);
  h_f_ext_off.setZero();

  data.SetExternalForce(h_f_ext_on);

  auto maybe_release_force = [&](int step) {
    if (force_release_step >= 0 && step == force_release_step) {
      data.SetExternalForce(h_f_ext_off);
      std::cout << "Released tip force at step " << step << std::endl;
    }
  };

  data.Setup(L, W, H, Quadrature::gauss_xi_m_6, Quadrature::gauss_xi_3,
             Quadrature::gauss_eta_2, Quadrature::gauss_zeta_2,
             Quadrature::weight_xi_m_6, Quadrature::weight_xi_3,
             Quadrature::weight_eta_2, Quadrature::weight_zeta_2, h_x12, h_y12,
             h_z12, h_element_connectivity);

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
    ANCFCPUUtils::VisualizationUtils::ExportANCF3243ToVTU(
        x12, y12, z12, h_element_connectivity, W, H, oss.str(), &vm);
  };

  if (want_vtu(0)) {
    Eigen::VectorXd x12, y12, z12;
    data.RetrievePositionToCPU(x12, y12, z12);
    export_vtu(0, x12, y12, z12);
  }

  const int tip_idx = tip_coef;
  std::vector<double> tip_z_history;
  if (write_csv) {
    tip_z_history.reserve(static_cast<size_t>(steps));
  }

  // The beam is 200 m by 0.1 m by default, so its section is drawn larger,
  // which is exact as positions are linear across the section
  const double section_scale =
      cli.GetDouble("section_scale") > 0.0
          ? cli.GetDouble("section_scale")
          : std::max(1.0, 0.02 * beam_length / std::max(W, H));
  char section_note[64] = "";
  if (section_scale != 1.0) {
    std::snprintf(section_note, sizeof(section_note), ", section x%g",
                  section_scale);
  }
  LiveView::LiveViewerOptions view_options = LiveView::ViewerOptions(
      cli, std::string("ANCF3243 beam (vbd") + section_note + ")");
  LiveView::LiveViewer viewer(view_options);
  viewer.AddSurface(
      "Beam",
      LiveView::ANCF3243Surface(h_x12, h_y12, h_z12, h_element_connectivity, L,
                                section_scale * W, section_scale * H),
      data.GetX12DevicePtr(), data.GetY12DevicePtr(), data.GetZ12DevicePtr());

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
        tip_z_history.push_back(z12(tip_idx));
      }
      if (do_vtu) {
        export_vtu(out_step, x12, y12, z12);
      }
    }
  });

  if (write_csv) {
    std::string out_path = cli.GetString("csv");
    if (out_path.empty()) {
      out_path = "tip_z_history_ancf3243_vbd.csv";
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
