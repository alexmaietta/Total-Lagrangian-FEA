#pragma once

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "../../lib_utils/cli_utils.h"
#include "../../lib_utils/mesh_manager.h"
#include "../../lib_utils/surface_trimesh.h"

namespace LiveView {

/**
 * Options for LiveViewer
 */
struct LiveViewerOptions {
  std::string title = "TL-FEA";
  unsigned width    = 1280;
  unsigned height   = 720;
  bool offscreen    = false;  // Render without a window (headless runs)
  bool show_ground  = true;   // A checkered ground under the surfaces
  // The viewer's own UI: its menus, toolbar, HUD, axis indicator and panels
  // (Stage, Properties, Info, Log, Profiler, Memory, Capture). Off leaves the
  // view with the colorbar, the status line and the timeline, whose settings
  // can turn it back on.
  bool show_viewer_ui = false;
  // Field the surfaces are colored by (displacement, or the node scalar given
  // to AddSurface), and the value at the top of the colormap. Zero picks a
  // tenth of the size of the surfaces at rest, for displacement.
  std::string color_label = "Displacement";
  std::string color_units = "m";
  float color_max         = 0.0f;
  // Write the last frame to this PNG file after Run or Play (implies
  // offscreen)
  std::string screenshot;
  // Memory for the frame history. Frames past the device budget move to
  // pinned host memory, and the oldest are dropped past the host budget.
  size_t history_device_mb = 1024;
  size_t history_host_mb   = 4096;
};

/**
 * A triangle surface whose vertices are fixed weighted sums of solver
 * coefficients. Vertex i is the sum over k of weights[i * terms + k] times
 * coefficient coefs[i * terms + k] of the solver's position arrays: a node
 * position for FEAT10 (one term of weight 1), and shape function values
 * times nodal coefficients for ANCF (see ANCFSurfaces.h).
 */
struct LinearSurface {
  int terms = 1;
  std::vector<int> coefs;             // (n_vertices * terms)
  std::vector<float> weights;         // (n_vertices * terms)
  std::vector<Eigen::Vector3d> rest;  // Vertex positions at rest
  std::vector<Eigen::Vector3i> triangles;
};

/**
 * Add the options every live demo takes (--offscreen, --screenshot,
 * --ground, --viewer_ui).
 * @param cli Command line to add them to
 * @param show_ground Whether the ground shows unless --ground says otherwise
 */
void AddViewerOptions(ANCFCPUUtils::Cli& cli, bool show_ground = true);

/**
 * Viewer options from the parsed command line.
 * @param cli Command line with AddViewerOptions applied
 * @param title Window title
 */
LiveViewerOptions ViewerOptions(const ANCFCPUUtils::Cli& cli,
                                const std::string& title);

/**
 * Live view of a simulation through Chrono Batch Render
 *
 * Each surface is a triangle mesh whose vertices follow solver nodes. Run
 * advances the simulation on a worker thread while the calling thread draws
 * the viewer, so the UI stays responsive however long a step takes. After
 * every step, one CUDA kernel gathers the surface nodes from the solver's
 * device arrays into a new frame of a history kept on the GPU, and the
 * drawing thread copies the frame on screen into the renderer's own memory.
 *
 * The history works like a recording. A timeline under the view follows the
 * newest frame while the simulation runs, and can be scrubbed, stepped,
 * played and looped over every frame recorded so far while the simulation
 * keeps adding frames. Keys are Space (play or pause playback), the left and
 * right arrows (step a frame), Home (first frame), End (follow the newest
 * frame), L (loop) and P (pause the simulation).
 *
 *     LiveView::LiveViewer viewer(LiveView::ViewerOptions(cli, "Bunny"));
 *     viewer.AddSurface("Bunny", nodes, elements, instance,
 *                       data.GetX12DevicePtr(), data.GetY12DevicePtr(),
 *                       data.GetZ12DevicePtr());
 *     viewer.Run(n_steps, [&](int step) { solver.Solve(); });
 *
 * Play shows recorded frames the same way, with a loader in place of the
 * solver (see lib_bin/live_view/test_vtk_playback).
 */
class LiveViewer {
 public:
  explicit LiveViewer(const LiveViewerOptions& options = {});
  ~LiveViewer();

  LiveViewer(const LiveViewer&)            = delete;
  LiveViewer& operator=(const LiveViewer&) = delete;

  /**
   * Add a surface whose vertex i follows node mesh.global_node_ids[i], with
   * mesh.vertices as the rest positions. Must be called before Run.
   * @param name Name of the surface in the scene
   * @param mesh Surface triangles (e.g. from
   * ANCFCPUUtils::ExtractSurfaceTriMesh)
   * @param d_x12 Node x positions on the device (one per node)
   * @param d_y12 Node y positions on the device (one per node)
   * @param d_z12 Node z positions on the device (one per node)
   * @param d_scalar Node scalar to color by on the device (one per node), or
   * nullptr to color by displacement
   */
  void AddSurface(const std::string& name,
                  const ANCFCPUUtils::SurfaceTriMesh& mesh, const double* d_x12,
                  const double* d_y12, const double* d_z12,
                  const double* d_scalar = nullptr);

  /**
   * Add a surface whose vertices are weighted sums of solver coefficients.
   * @param name Name of the surface in the scene
   * @param surface Vertices, their coefficients and weights, and triangles
   * @param d_x12 Coefficient x components on the device
   * @param d_y12 Coefficient y components on the device
   * @param d_z12 Coefficient z components on the device
   * @param d_scalar Scalar to color by on the device (indexed like the
   * coefficients, read at each vertex's first one), or nullptr to color by
   * displacement
   */
  void AddSurface(const std::string& name, const LinearSurface& surface,
                  const double* d_x12, const double* d_y12, const double* d_z12,
                  const double* d_scalar = nullptr);

  /**
   * Add the boundary surface of a FEAT10 mesh instance, extracted from the
   * rest configuration.
   * @param name Name of the surface in the scene
   * @param nodes Rest positions of all nodes (n_nodes x 3)
   * @param elements T10 connectivity of all elements (n_elems x 10)
   * @param instance Nodes and elements of this mesh within the arrays
   * @param d_x12 Node x positions on the device (one per node)
   * @param d_y12 Node y positions on the device (one per node)
   * @param d_z12 Node z positions on the device (one per node)
   * @param d_scalar Node scalar to color by on the device (one per node), or
   * nullptr to color by displacement
   */
  void AddSurface(const std::string& name, const Eigen::MatrixXd& nodes,
                  const Eigen::MatrixXi& elements,
                  const ANCFCPUUtils::MeshInstance& instance,
                  const double* d_x12, const double* d_y12, const double* d_z12,
                  const double* d_scalar = nullptr);

  /**
   * Call step(0), step(1), ... on a worker thread and record a frame after
   * every step, following the newest frame. With a window, returns once it is
   * closed (the step in progress finishing first); offscreen, returns after
   * the last step is drawn, and writes options.screenshot if set.
   * @param n_steps Number of steps, or 0 to run until the window is closed
   * @param step Advances the simulation by one step
   */
  void Run(int n_steps, const std::function<void(int)>& step);

  /**
   * Play back saved frames. load(frame) runs on a worker thread and must put
   * that frame's node positions (and scalar) into the device arrays given to
   * AddSurface. Frames are recorded into the history as they load, and the
   * timeline plays them from the first. With a window, returns once it is
   * closed; offscreen, returns after the last frame is drawn (without
   * looping), and writes options.screenshot if set.
   * @param n_frames Number of frames
   * @param fps Frames shown per second
   * @param loop Start over after the last frame
   * @param load Loads one frame into the device arrays
   */
  void Play(int n_frames, double fps, bool loop,
            const std::function<void(int)>& load);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace LiveView
