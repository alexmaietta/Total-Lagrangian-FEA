/**
 * VTK Playback Demo
 *
 * Plays back FEAT10 frames in Chrono Batch Render, either the .vtu files a
 * demo wrote (e.g. test_feat10_tire_newton) or the legacy .vtk files of the
 * FEniCS validation runs. Frames can be colored by displacement and by one
 * field of the files, von Mises stress when they have it. They are read with
 * the small ASCII VTK reader in lib_src/live_view and loaded on a worker
 * thread, so the viewer stays responsive. Space pauses, the left and right
 * arrows step a frame, Home restarts and L toggles looping.
 *
 * Example:
 *   bazel run //lib_bin/live_view:test_vtk_playback -- \
 *     --frames='output/tire_newton_res0_step_*.vtu' --loop
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../../lib_src/live_view/LiveViewer.h"
#include "../../lib_src/live_view/VtkReader.h"
#include "../../lib_utils/cli_utils.h"
#include "../../lib_utils/cuda_utils.h"

int main(int argc, char** argv) {
  ANCFCPUUtils::Cli cli(argv[0]);
  cli.SetDescription(
      "Play back FEAT10 .vtu or .vtk frames in Chrono Batch Render.");
  cli.AddString("frames", "", "frames to play, as a glob pattern (quote it)");
  cli.AddString("color", "auto",
                "field of the files to color by besides displacement "
                "(von_mises_stress, displacement_magnitude, another field, or "
                "displacement for none; auto picks von_mises_stress when "
                "present)");
  cli.AddDouble("color_max", 0.0,
                "value at the top of the colormap for that field (0 fits the "
                "frames loaded)");
  cli.AddDouble("fps", 30.0, "frames shown per second");
  cli.AddBool("loop", false, "start over after the last frame");
  LiveView::AddViewerOptions(cli);
  std::string cli_err;
  if (!cli.Parse(argc, argv, &cli_err) || cli.HelpRequested() ||
      cli.GetString("frames").empty()) {
    if (!cli_err.empty()) {
      std::cerr << cli_err << "\n\n";
    }
    cli.PrintUsage(std::cout);
    return cli.HelpRequested() ? 0 : 1;
  }

  // bazel run starts in the runfiles tree, so resolve relative patterns
  // against the directory it was invoked from
  std::string pattern = cli.GetString("frames");
  if (pattern[0] != '/' && std::getenv("BUILD_WORKING_DIRECTORY")) {
    pattern =
        std::string(std::getenv("BUILD_WORKING_DIRECTORY")) + "/" + pattern;
  }
  const std::vector<std::string> files = LiveView::ListFrames(pattern);
  if (files.empty()) {
    std::cerr << "No files match " << pattern << std::endl;
    return 1;
  }

  std::string color = cli.GetString("color");
  if (color == "auto") {
    LiveView::VtkPiece first;
    std::string error;
    if (!LiveView::ReadVtkPiece(files[0], first, &error)) {
      std::cerr << error << std::endl;
      return 1;
    }
    color = first.CellArray("von_mises_stress") ||
                    first.PointArray("von_mises_stress")
                ? "von_mises_stress"
                : "displacement";
  }
  const std::string scalar = color == "displacement" ? "" : color;
  LiveView::VtkSequence sequence(files, scalar);
  const int n_nodes = sequence.NumNodes();
  std::printf("%d frames, %d nodes, %d elements (%s ... %s)\n", sequence.Size(),
              n_nodes, sequence.NumElements(), files.front().c_str(),
              files.back().c_str());

  // Device arrays standing in for the solver's, filled one frame at a time
  double *d_x = nullptr, *d_y = nullptr, *d_z = nullptr, *d_scalar = nullptr;
  HANDLE_ERROR(cudaMalloc(&d_x, n_nodes * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&d_y, n_nodes * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&d_z, n_nodes * sizeof(double)));
  if (!scalar.empty()) {
    HANDLE_ERROR(cudaMalloc(&d_scalar, n_nodes * sizeof(double)));
  }

  LiveView::LiveViewer viewer(LiveView::ViewerOptions(cli, "TL-FEA playback"));
  viewer.AddSurface("Mesh", sequence.RestNodes(), sequence.Elements(),
                    {0, 0, n_nodes, sequence.NumElements(), "mesh"}, d_x, d_y,
                    d_z);
  if (!scalar.empty()) {
    const bool vm = scalar == "von_mises_stress";
    viewer.AddField({vm ? "von Mises stress" : scalar,
                     vm                                   ? "Pa"
                     : scalar == "displacement_magnitude" ? "m"
                                                          : "",
                     LiveView::NodeScalar(d_scalar), 0.0f,
                     float(cli.GetDouble("color_max"))});
  }
  viewer.AddField({"Displacement", "m", LiveView::Displacement()});

  std::vector<double> xyz, values;
  viewer.Play(sequence.Size(), cli.GetDouble("fps"), cli.GetBool("loop"),
              [&](int frame) {
                sequence.Frame(frame, xyz, values);
                const size_t bytes = n_nodes * sizeof(double);
                HANDLE_ERROR(
                    cudaMemcpy(d_x, xyz.data(), bytes, cudaMemcpyHostToDevice));
                HANDLE_ERROR(cudaMemcpy(d_y, xyz.data() + n_nodes, bytes,
                                        cudaMemcpyHostToDevice));
                HANDLE_ERROR(cudaMemcpy(d_z, xyz.data() + 2 * n_nodes, bytes,
                                        cudaMemcpyHostToDevice));
                if (d_scalar) {
                  HANDLE_ERROR(cudaMemcpy(d_scalar, values.data(), bytes,
                                          cudaMemcpyHostToDevice));
                }
              });

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(d_scalar);
  return 0;
}
