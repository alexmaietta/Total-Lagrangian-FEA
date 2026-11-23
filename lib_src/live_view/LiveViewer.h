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
  std::string title   = "TL-FEA";
  unsigned width      = 1280;
  unsigned height     = 720;
  bool offscreen      = false;  // No window
  bool show_ground    = true;
  bool show_viewer_ui = false;  // Menus, toolbar, HUD and panels
  std::string field;            // Shown first, or empty for the first added
  std::string screenshot;       // Last frame's PNG (implies offscreen)
  // Saved frames: every capture_every-th and the last, the run waiting on
  // saving rather than dropping frames
  std::string video;  // H.264 through ffmpeg
  std::string png_dir;
  int capture_every = 1;
  double video_fps  = 30.0;
  // Camera about the model's center, in degrees and model sizes
  double camera_azimuth   = -51.3;  // From +x toward +y
  double camera_elevation = 19.4;
  double camera_distance  = 0.0;  // 0 frames the model
  // Frame history, spilling to pinned host memory, then dropping the oldest
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
  std::vector<Eigen::Vector3d> rest;  // At rest
  std::vector<Eigen::Vector3i> triangles;
  // Elements of each vertex (CSR), for element fields; empty if unknown
  std::vector<int> element_offsets;  // (n_vertices + 1)
  std::vector<int> elements;
};

/**
 * Which value of a vector a field shows
 */
enum class Component { Magnitude, X, Y, Z };

/**
 * What a field kernel is given for one surface, all on the device
 */
struct FieldContext {
  std::string surface;  // Name given to AddSurface
  int n_vertices       = 0;
  int terms            = 1;
  const int* coefs     = nullptr;  // (n_vertices * terms)
  const float* weights = nullptr;  // (n_vertices * terms)
  const float* points  = nullptr;  // This frame, [x0,y0,z0, ...]
  const float* rest    = nullptr;  // [x0,y0,z0, ...]
};

/**
 * Computes a field's value at each vertex of a surface. Runs on the
 * simulation thread after each step, and must write ctx.n_vertices floats to
 * values, queued on stream.
 */
using FieldKernel = std::function<void(const FieldContext& ctx, float* values,
                                       cudaStream_t stream)>;

/**
 * Where a field's values come from. Made by the functions below.
 */
struct FieldSource {
  enum class Kind {
    Displacement,
    NodeScalar,
    NodeVector,
    ElementScalar,
    Kernel
  };
  Kind kind           = Kind::Displacement;
  Component component = Component::Magnitude;
  const double* x     = nullptr;  // Scalar or x
  const double* y     = nullptr;
  const double* z     = nullptr;
  int stride          = 1;  // Between coefficients
  FieldKernel kernel;
};

/**
 * Displacement of each vertex from its rest position.
 */
FieldSource Displacement(Component component = Component::Magnitude);

/**
 * A scalar per coefficient on the device, read at each vertex's first
 * coefficient (its node for FEAT10).
 */
FieldSource NodeScalar(const double* d_values);

/**
 * A vector per coefficient on the device, interpolated to each vertex like
 * its position (so a vertex's velocity from the coefficient velocities).
 * @param d_xyz Interleaved components, [x0,y0,z0, x1,y1,z1, ...]
 */
FieldSource NodeVector(const double* d_xyz,
                       Component component = Component::Magnitude);

/**
 * A vector per coefficient on the device, with each component in its own
 * array (like the solver's positions).
 */
FieldSource NodeVector(const double* d_x, const double* d_y, const double* d_z,
                       Component component = Component::Magnitude);

/**
 * A scalar per element on the device (such as von Mises stress), averaged
 * over the elements each vertex lies in. Needs surfaces that know their
 * elements (LinearSurface::elements).
 */
FieldSource ElementScalar(const double* d_values);

/**
 * Values computed by a kernel of your own.
 */
FieldSource Kernel(FieldKernel kernel);

/**
 * A field the surfaces can be colored by. Every field is recorded with every
 * frame, so the one on screen can change at any time, past frames included.
 */
struct Field {
  std::string name;
  std::string units;
  FieldSource source;
  // Colormap range; max <= min fits the values recorded so far
  float min = 0.0f;
  float max = 0.0f;
  // Run after each step, before recording (e.g. data.ComputeVonMises())
  std::function<void()> update;
};

/**
 * Add the options of LiveViewerOptions every live demo takes (--offscreen,
 * --ground, --field, --video, --azimuth and so on).
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
 *     viewer.AddField({"Displacement", "m", LiveView::Displacement()});
 *     viewer.AddField({"von Mises stress", "Pa",
 *                      LiveView::ElementScalar(data.GetVonMisesDevicePtr()),
 *                      0, 0, [&] { data.ComputeVonMises(); }});
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
   */
  void AddSurface(const std::string& name,
                  const ANCFCPUUtils::SurfaceTriMesh& mesh, const double* d_x12,
                  const double* d_y12, const double* d_z12);

  /**
   * Add a surface whose vertices are weighted sums of solver coefficients.
   * @param name Name of the surface in the scene
   * @param surface Vertices, their coefficients and weights, and triangles
   * @param d_x12 Coefficient x components on the device
   * @param d_y12 Coefficient y components on the device
   * @param d_z12 Coefficient z components on the device
   */
  void AddSurface(const std::string& name, const LinearSurface& surface,
                  const double* d_x12, const double* d_y12,
                  const double* d_z12);

  /**
   * Add the boundary surface of a FEAT10 mesh instance, extracted from the
   * rest configuration, knowing the elements each vertex lies in.
   * @param name Name of the surface in the scene
   * @param nodes Rest positions of all nodes (n_nodes x 3)
   * @param elements T10 connectivity of all elements (n_elems x 10)
   * @param instance Nodes and elements of this mesh within the arrays
   * @param d_x12 Node x positions on the device (one per node)
   * @param d_y12 Node y positions on the device (one per node)
   * @param d_z12 Node z positions on the device (one per node)
   */
  void AddSurface(const std::string& name, const Eigen::MatrixXd& nodes,
                  const Eigen::MatrixXi& elements,
                  const ANCFCPUUtils::MeshInstance& instance,
                  const double* d_x12, const double* d_y12,
                  const double* d_z12);

  /**
   * Add a field to color by, on every surface. Must be called before Run.
   * Without any, the surfaces are colored by displacement.
   */
  void AddField(Field field);

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
   * that frame's node positions (and field arrays) into the device arrays
   * given to AddSurface and AddField. Frames are recorded into the history as
   * they load, and the timeline plays them from the first. With a window,
   * returns once it is closed; offscreen, returns after the last frame is drawn
   * (without looping), and writes options.screenshot if set.
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
