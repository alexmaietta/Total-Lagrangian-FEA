/**
 * FEAT10 Double Pendulum Spherical-Joint Live Demo
 *
 * Runs the spherical-joint double pendulum demo
 * (test_feat10_double_pendulum_spherical) and shows both links live in
 * Chrono Batch Render instead of exporting VTU files. The upper link is
 * attached to the world through a spherical joint, and the lower link to
 * the upper link through a second one. Press P to pause the solver.
 *
 * Example:
 *   bazel run //lib_bin/live_view:test_feat10_double_pendulum_spherical_live
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/live_view/LiveViewer.h"
#include "../../lib_src/solvers/FEAT10ConstraintManager.h"
#include "../../lib_src/solvers/SyncedNewton.cuh"
#include "../../lib_utils/cli_utils.h"
#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/mesh_manager.h"
#include "../../lib_utils/quadrature_utils.h"

namespace {

constexpr double kPi           = 3.14159265358979323846;
constexpr double kGravity      = -9.81;
constexpr double kDt           = 2e-4;
constexpr int kNumStepsDefault = 40000;

constexpr double kBeamLength           = 0.5;
constexpr double kUpperHingeLocalZ     = 0.0;
constexpr double kLowerHingeLocalZ     = kBeamLength;
constexpr double kLowerLinkHingeLocalZ = 0.0;
constexpr double kUpperAngleDeg        = 35.0;
constexpr double kLowerAngleDeg        = -25.0;
constexpr double kUpperOutOfPlaneDeg   = 18.0;
constexpr double kLowerOutOfPlaneDeg   = -27.0;

const SolidMaterialProperties kLinkMaterial =
    SolidMaterialProperties::SVK(5.0e6,   // E: pendulum links
                                 0.30,    // nu
                                 1200.0,  // rho0
                                 1.0e3,   // eta_damp
                                 1.0e3    // lambda_damp
    );

double DegToRad(double angle_deg) {
  return angle_deg * kPi / 180.0;
}

Eigen::Vector3d TransformPoint(const Eigen::Matrix4d& transform,
                               const Eigen::Vector3d& point) {
  const Eigen::Vector4d p_h(point.x(), point.y(), point.z(), 1.0);
  const Eigen::Vector4d mapped = transform * p_h;
  return mapped.head<3>();
}

Eigen::Matrix4d MakeBeamTransform(double angle_x, double angle_y,
                                  const Eigen::Vector3d& hinge_point,
                                  double hinge_local_z) {
  // Start from the planar Y rotation used by the revolute demo, then add an
  // X tilt so the spherical-joint motion is initialized in full 3D.
  const Eigen::Matrix4d rotation =
      ANCFCPUUtils::rotationX(angle_x) * ANCFCPUUtils::rotationY(angle_y);
  const Eigen::Vector3d rotated_hinge =
      TransformPoint(rotation, Eigen::Vector3d(0.0, 0.0, hinge_local_z));
  return ANCFCPUUtils::translation(hinge_point.x() - rotated_hinge.x(),
                                   hinge_point.y() - rotated_hinge.y(),
                                   hinge_point.z() - rotated_hinge.z()) *
         rotation;
}

FEAT10ConstraintManager::ElementRange MakeElementRange(
    const ANCFCPUUtils::MeshInstance& instance) {
  return FEAT10ConstraintManager::ElementRange{
      instance.element_offset, instance.element_offset + instance.num_elements};
}

std::vector<double> ComputeLumpedMass(GPU_FEAT10_Data& data) {
  std::vector<int> offsets;
  std::vector<int> columns;
  std::vector<double> values;
  data.RetrieveMassCSRToCPU(offsets, columns, values);

  std::vector<double> lumped_mass(static_cast<size_t>(data.get_n_coef()), 0.0);
  if (offsets.size() != static_cast<size_t>(data.get_n_coef() + 1)) {
    std::fill(lumped_mass.begin(), lumped_mass.end(), 1.0);
    return lumped_mass;
  }

  for (int row = 0; row < data.get_n_coef(); ++row) {
    double row_sum = 0.0;
    for (int idx = offsets[static_cast<size_t>(row)];
         idx < offsets[static_cast<size_t>(row + 1)]; ++idx) {
      row_sum += values[static_cast<size_t>(idx)];
    }
    lumped_mass[static_cast<size_t>(row)] = row_sum;
  }
  return lumped_mass;
}

void AppendGravityForInstance(Eigen::VectorXd* h_f_ext,
                              const std::vector<double>& lumped_mass,
                              const ANCFCPUUtils::MeshInstance& instance,
                              double gravity) {
  for (int local_node = 0; local_node < instance.num_nodes; ++local_node) {
    const int global_node = instance.node_offset + local_node;
    (*h_f_ext)(3 * global_node + 2) +=
        lumped_mass[static_cast<size_t>(global_node)] * gravity;
  }
}

}  // namespace

int main(int argc, char** argv) {
  ANCFCPUUtils::Cli cli(argv[0]);
  cli.SetDescription(
      "FEAT10 double pendulum (spherical joints) shown live in Chrono Batch "
      "Render.");
  cli.AddInt("steps", kNumStepsDefault, "steps to run");
  LiveView::AddViewerOptions(cli);
  std::string cli_err;
  if (!cli.Parse(argc, argv, &cli_err) || cli.HelpRequested()) {
    if (!cli_err.empty()) {
      std::cerr << cli_err << "\n\n";
    }
    cli.PrintUsage(std::cout);
    return cli.HelpRequested() ? 0 : 1;
  }
  const int max_steps = cli.GetInt("steps");

  std::cout << "========================================\n";
  std::cout << "FEAT10 Double Pendulum Spherical-Joint Demo\n";
  std::cout << "========================================\n";
  std::cout << "steps=" << max_steps << "\n";

  const std::string mesh_prefix =
      "data/meshes/T10/double_pendulum/pendulum_beam.1";

  ANCFCPUUtils::MeshManager mesh_manager;
  const int mesh_upper = mesh_manager.LoadMesh(
      mesh_prefix + ".node", mesh_prefix + ".ele", "upper", kLinkMaterial);
  const int mesh_lower = mesh_manager.LoadMesh(
      mesh_prefix + ".node", mesh_prefix + ".ele", "lower", kLinkMaterial);
  if (mesh_upper < 0 || mesh_lower < 0) {
    std::cerr << "Failed to load pendulum beam meshes from " << mesh_prefix
              << std::endl;
    return 1;
  }

  const auto& inst_upper = mesh_manager.GetMeshInstance(mesh_upper);
  const auto& inst_lower = mesh_manager.GetMeshInstance(mesh_lower);

  const Eigen::Vector3d top_hinge(0.0, 0.0, 0.7);
  const double upper_angle        = kPi - DegToRad(kUpperAngleDeg);
  const double lower_angle        = kPi - DegToRad(kLowerAngleDeg);
  const double upper_out_of_plane = DegToRad(kUpperOutOfPlaneDeg);
  const double lower_out_of_plane = DegToRad(kLowerOutOfPlaneDeg);

  const Eigen::Matrix4d upper_transform = MakeBeamTransform(
      upper_out_of_plane, upper_angle, top_hinge, kUpperHingeLocalZ);
  mesh_manager.TransformMesh(mesh_upper, upper_transform);

  const Eigen::Vector3d lower_hinge = TransformPoint(
      upper_transform, Eigen::Vector3d(0.0, 0.0, kLowerHingeLocalZ));
  const Eigen::Matrix4d lower_transform = MakeBeamTransform(
      lower_out_of_plane, lower_angle, lower_hinge, kLowerLinkHingeLocalZ);
  mesh_manager.TransformMesh(mesh_lower, lower_transform);

  const Eigen::MatrixXd& all_nodes = mesh_manager.GetAllNodes();
  const Eigen::MatrixXi& all_elems = mesh_manager.GetAllElements();
  const int n_nodes                = mesh_manager.GetTotalNodes();
  const int n_elems                = mesh_manager.GetTotalElements();

  std::cout << "upper:   " << inst_upper.num_nodes << " nodes, "
            << inst_upper.num_elements << " elements\n";
  std::cout << "lower:   " << inst_lower.num_nodes << " nodes, "
            << inst_lower.num_elements << " elements\n";
  std::cout << "total:   " << n_nodes << " nodes, " << n_elems << " elements\n";
  std::cout << "top hinge:   [" << top_hinge.transpose() << "]\n";
  std::cout << "lower hinge: [" << lower_hinge.transpose() << "]\n";
  std::cout << "upper out-of-plane tilt: " << kUpperOutOfPlaneDeg << " deg\n";
  std::cout << "lower out-of-plane tilt: " << kLowerOutOfPlaneDeg << " deg\n";

  GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);
  gpu_t10_data.Initialize();

  Eigen::VectorXd h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
  for (int i = 0; i < n_nodes; ++i) {
    h_x12(i) = all_nodes(i, 0);
    h_y12(i) = all_nodes(i, 1);
    h_z12(i) = all_nodes(i, 2);
  }

  gpu_t10_data.Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y,
                     Quadrature::tet5pt_z, Quadrature::tet5pt_weights, h_x12,
                     h_y12, h_z12, all_elems);
  gpu_t10_data.ApplyMaterialsFromMeshManager(mesh_manager);
  gpu_t10_data.CalcDnDuPre();
  gpu_t10_data.CalcMassMatrix();

  const std::vector<double> lumped_mass = ComputeLumpedMass(gpu_t10_data);
  Eigen::VectorXd h_f_ext               = Eigen::VectorXd::Zero(n_nodes * 3);
  AppendGravityForInstance(&h_f_ext, lumped_mass, inst_upper, kGravity);
  AppendGravityForInstance(&h_f_ext, lumped_mass, inst_lower, kGravity);
  gpu_t10_data.SetExternalForce(h_f_ext);

  FEAT10ConstraintManager constraint_manager(&gpu_t10_data);
  constraint_manager.AddSphericalJointToWorld(MakeElementRange(inst_upper),
                                              top_hinge);
  constraint_manager.AddSphericalJoint(
      MakeElementRange(inst_upper), MakeElementRange(inst_lower), lower_hinge);
  constraint_manager.Finalize();

  gpu_t10_data.CalcConstraintData();
  gpu_t10_data.ConvertToCSR_ConstraintJacT();
  gpu_t10_data.BuildConstraintJacobianCSR();
  gpu_t10_data.CalcP();
  gpu_t10_data.CalcInternalForce();

  SyncedNewtonParams params = {1e-4, 1e-4, 1e-8, 1e8, 8, 10, kDt, false};
  SyncedNewtonSolver solver(&gpu_t10_data, gpu_t10_data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);
  solver.AnalyzeHessianSparsity();
  solver.SetFixedSparsityPattern(true);

  std::cout << "constraints: " << gpu_t10_data.get_n_constraint() << "\n";

  LiveView::LiveViewer viewer(LiveView::ViewerOptions(
      cli, "FEAT10 double pendulum (spherical joints)"));
  viewer.AddSurface("UpperLink", all_nodes, all_elems, inst_upper,
                    gpu_t10_data.GetX12DevicePtr(),
                    gpu_t10_data.GetY12DevicePtr(),
                    gpu_t10_data.GetZ12DevicePtr());
  viewer.AddSurface("LowerLink", all_nodes, all_elems, inst_lower,
                    gpu_t10_data.GetX12DevicePtr(),
                    gpu_t10_data.GetY12DevicePtr(),
                    gpu_t10_data.GetZ12DevicePtr());

  viewer.AddField({"Displacement", "m", LiveView::Displacement()});
  viewer.AddField({"von Mises stress", "Pa",
                   LiveView::ElementScalar(gpu_t10_data.GetVonMisesDevicePtr()),
                   0, 0, [&] { gpu_t10_data.ComputeVonMises(); }});
  viewer.AddField({"Velocity", "m/s",
                   LiveView::NodeVector(solver.GetVelocityGuessDevicePtr())});

  viewer.Run(max_steps, [&](int) { solver.Solve(); });

  gpu_t10_data.Destroy();
  return 0;
}
