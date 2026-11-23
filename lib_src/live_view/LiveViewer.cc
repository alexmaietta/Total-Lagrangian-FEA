#include "LiveViewer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <variant>
#include <vector>

#include "../../lib_utils/cuda_utils.h"
#include "../../lib_utils/surface_trimesh_extract.h"
#include "LiveViewerKernels.cuh"
#include "chrono_batch_render/app/viewer/viewer.h"
#include "chrono_batch_render/chrono_batch_render.h"
#include "chrono_batch_render/io/image.h"
#include "imgui.h"

namespace cbr = chrono::batch_render;

namespace LiveView {
namespace {

using Clock = std::chrono::steady_clock;

// Lucide icons in the viewer's font (chrono_batch_render/app/viewer/theme)
constexpr const char* kIconSettings    = "\xee\x85\x94";
constexpr const char* kIconPlay        = "\xee\x84\xbc";
constexpr const char* kIconPause       = "\xee\x84\xae";
constexpr const char* kIconSkipBack    = "\xee\x85\x9f";
constexpr const char* kIconSkipForward = "\xee\x85\xa0";
constexpr const char* kIconStepBack    = "\xee\x8f\xa9";
constexpr const char* kIconStepForward = "\xee\x8f\xaa";
constexpr const char* kIconRepeat      = "\xee\x85\x86";
constexpr const char* kIconSquare      = "\xee\x85\xa7";

constexpr const char* kViewport = "Simulation";

// Device address of a mapping's values, after checking that it is CUDA memory
// holding n_vertices float3 values.
float* MappedFloat3(const cbr::Mapping& mapping, int n_vertices,
                    const std::string& what) {
  const DLTensor values = mapping.DLPack();
  if (values.device.device_type != kDLCUDA) {
    throw std::runtime_error(
        "LiveViewer " + what + " is not CUDA memory (DLPack device type " +
        std::to_string(int(values.device.device_type)) + ")");
  }
  int64_t elements = 1;
  for (int d = 0; d < values.ndim; ++d) {
    elements *= values.shape[d];
  }
  const int64_t bytes = elements * values.dtype.bits * values.dtype.lanes / 8;
  if (bytes != int64_t(n_vertices) * 3 * int64_t(sizeof(float))) {
    throw std::runtime_error("LiveViewer " + what + " holds " +
                             std::to_string(bytes) + " bytes, expected " +
                             std::to_string(n_vertices * 12));
  }
  return reinterpret_cast<float*>(static_cast<char*>(values.data) +
                                  values.byte_offset);
}

/**
 * Recorded frames of fixed size, in slots carved from blocks of device memory
 * up to a budget, then of pinned host memory. When the device budget is used
 * up, the oldest frame on the device moves to the host; when the host budget
 * is used up too, the oldest frame is dropped.
 */
class FrameHistory {
 public:
  ~FrameHistory() {
    for (void* block : device_blocks_) {
      cudaFree(block);
    }
    for (void* block : host_blocks_) {
      cudaFreeHost(block);
    }
  }

  void Configure(size_t frame_bytes, size_t device_budget, size_t host_budget) {
    frame_bytes_   = frame_bytes;
    device_budget_ = device_budget;
    host_budget_   = host_budget;
  }

  // Device slot for the next frame (making room if needed)
  void* Acquire() {
    void* slot = NewSlot(true);
    while (slot == nullptr) {
      auto oldest = OldestOnDevice();
      if (oldest == frames_.end()) {
        throw std::runtime_error("LiveViewer history budget is too small");
      }
      void* host = NewSlot(false);
      if (host == nullptr && !frames_.front().device) {
        // Host budget used up; drop the oldest frame for its slot
        WaitForReader(first_);
        free_host_.push_back(frames_.front().data);
        frames_.pop_front();
        ++first_;
        host   = NewSlot(false);
        oldest = OldestOnDevice();
      }
      if (host == nullptr) {
        // No host memory at all; drop the oldest frame outright
        WaitForReader(first_);
        free_device_.push_back(frames_.front().data);
        frames_.pop_front();
        ++first_;
      } else {
        WaitForReader(first_ + int(oldest - frames_.begin()));
        HANDLE_ERROR(cudaMemcpy(host, oldest->data, frame_bytes_,
                                cudaMemcpyDeviceToHost));
        free_device_.push_back(oldest->data);
        *oldest = {host, false};
      }
      slot = NewSlot(true);
    }
    return slot;
  }

  void Commit(void* slot) {
    frames_.push_back({slot, true});
  }

  // A frame being read until 'done' completes, which Acquire waits for
  // before moving or dropping it
  void SetReader(int frame, cudaEvent_t done) {
    read_frame_ = frame;
    read_done_  = done;
  }

  int First() const {
    return first_;
  }

  // One past the newest frame
  int End() const {
    return first_ + int(frames_.size());
  }

  const char* Data(int frame) const {
    return static_cast<const char*>(frames_.at(size_t(frame - first_)).data);
  }

  size_t DeviceBytes() const {
    return device_bytes_;
  }

  size_t HostBytes() const {
    return host_bytes_;
  }

 private:
  struct Entry {
    void* data;
    bool device;
  };

  static constexpr size_t kSlotsPerBlock = 16;

  void WaitForReader(int frame) {
    if (frame == read_frame_) {
      HANDLE_ERROR(cudaEventSynchronize(read_done_));
      read_frame_ = -1;
    }
  }

  std::deque<Entry>::iterator OldestOnDevice() {
    return std::find_if(frames_.begin(), frames_.end(),
                        [](const Entry& e) { return e.device; });
  }

  // A free slot, or nullptr when the budget allows no new block
  void* NewSlot(bool device) {
    std::vector<void*>& free = device ? free_device_ : free_host_;
    if (free.empty()) {
      size_t& used        = device ? device_bytes_ : host_bytes_;
      const size_t budget = device ? device_budget_ : host_budget_;
      size_t slots        = kSlotsPerBlock;
      while (slots > 1 && used + slots * frame_bytes_ > budget) {
        slots /= 2;
      }
      // Always allow one device slot, so a tiny budget still records
      if (used + slots * frame_bytes_ > budget && !(device && used == 0)) {
        return nullptr;
      }
      void* block = nullptr;
      if (device) {
        HANDLE_ERROR(cudaMalloc(&block, slots * frame_bytes_));
        device_blocks_.push_back(block);
      } else {
        HANDLE_ERROR(cudaMallocHost(&block, slots * frame_bytes_));
        host_blocks_.push_back(block);
      }
      used += slots * frame_bytes_;
      for (size_t i = 0; i < slots; ++i) {
        free.push_back(static_cast<char*>(block) + i * frame_bytes_);
      }
    }
    void* slot = free.back();
    free.pop_back();
    return slot;
  }

  size_t frame_bytes_   = 0;
  size_t device_budget_ = 0;
  size_t host_budget_   = 0;
  std::deque<Entry> frames_;
  int first_ = 0;
  std::vector<void*> free_device_;
  std::vector<void*> free_host_;
  std::vector<void*> device_blocks_;
  std::vector<void*> host_blocks_;
  size_t device_bytes_   = 0;
  size_t host_bytes_     = 0;
  int read_frame_        = -1;
  cudaEvent_t read_done_ = nullptr;
};

}  // namespace

class LiveViewer::Impl {
 public:
  struct Surface {
    std::string name;
    std::string path;
    int n_vertices         = 0;
    int terms              = 1;
    const double* d_x12    = nullptr;  // Solver's
    const double* d_y12    = nullptr;
    const double* d_z12    = nullptr;
    int* d_coefs           = nullptr;  // (n_vertices * terms)
    float* d_weights       = nullptr;  // (n_vertices * terms)
    float* d_rest          = nullptr;  // (n_vertices * 3)
    int* d_element_offsets = nullptr;  // CSR, or nullptr
    int* d_elements        = nullptr;
    size_t points_offset   = 0;  // In a frame
    size_t values_offset   = 0;  // Fields follow one another
  };

  struct FieldState {
    Field field;
    float scale = 1.0f;  // Of the colormap range
    // Recorded so far (under mutex_)
    float lo = FLT_MAX;
    float hi = -FLT_MAX;
  };

  explicit Impl(const LiveViewerOptions& options) : options_(options) {
    if (!options_.screenshot.empty()) {
      options_.offscreen = true;
    }
    // The CUDA backend can't present to windows. Vulkan shares its memory
    // with CUDA, so the mappings are still CUDA device pointers.
    cbr::device::DeviceDesc desc;
    desc.backend      = cbr::device::Backend::Vulkan;
    desc.presentation = !options_.offscreen;
    device_           = cbr::device::Device::Create(desc);
    scene_            = cbr::Scene::Create(*device_);
    // Non-blocking, so copies into the renderer never wait for solver work
    // queued on the legacy default stream; at the highest priority, so the
    // colorize kernel runs ahead of queued solver kernels
    int least = 0, greatest = 0;
    HANDLE_ERROR(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    HANDLE_ERROR(cudaStreamCreateWithPriority(&copy_stream_,
                                              cudaStreamNonBlocking, greatest));
    HANDLE_ERROR(
        cudaEventCreateWithFlags(&publish_done_, cudaEventDisableTiming));
  }

  ~Impl() {
    // The renderer may still wait on the copy stream, so release it first
    cudaStreamSynchronize(copy_stream_);
    viewer_.reset();
    scene_  = {};
    device_ = {};
    for (Surface& s : surfaces_) {
      cudaFree(s.d_coefs);
      cudaFree(s.d_weights);
      cudaFree(s.d_rest);
      cudaFree(s.d_element_offsets);
      cudaFree(s.d_elements);
    }
    cudaFree(d_values_);
    cudaFree(d_ranges_);
    cudaEventDestroy(publish_done_);
    cudaStreamDestroy(copy_stream_);
  }

  void AddSurface(const std::string& name, const LinearSurface& linear,
                  const double* d_x12, const double* d_y12,
                  const double* d_z12);
  void AddField(Field field) {
    if (viewer_) {
      throw std::runtime_error("LiveViewer fields must be added before Run");
    }
    fields_.push_back({std::move(field)});
  }
  void Run(int n_steps, const std::function<void(int)>& step);
  void Play(int n_frames, double fps, bool loop,
            const std::function<void(int)>& load);

 private:
  enum class Mode { Simulation, Playback };

  double Size() const {
    return std::max({hi_[0] - lo_[0], hi_[1] - lo_[1], hi_[2] - lo_[2], 1e-6});
  }

  void Open();
  void Record(double step_ms);
  void RecordField(const FieldSource& source, const Surface& s,
                   const float* points, float* values);
  std::array<float, 2> ColorRange(int field);
  void UpdateColorbar();
  void SelectField(int field) {
    field_ = field;
    shown_ = -1;  // Recolor the frame on screen
  }
  bool Publish(int frame);
  void Drive(std::thread& producer);
  void UpdateTimeline();
  void DrawTimeline();
  void DrawRuler(float width);
  void DrawTimelineSettings();
  bool OnKey(const cbr::app::KeyEvent& event);
  void SavePng(const std::string& path);

  // Timeline actions (drawing thread)
  void Seek(int frame) {
    follow_  = false;
    playing_ = false;
    cursor_  = frame;
  }
  void StepFrames(int delta) {
    Seek(cursor_ + delta);
  }
  void TogglePlay();
  void Stop() {
    Seek(RangeStart());
  }
  void FollowNewest() {
    follow_  = true;
    playing_ = false;
  }
  // The range the ruler shows and playback stays in (the start and end time
  // codes), the whole timeline unless edited
  int RangeStart() const {
    return std::max(range_start_, 0);
  }
  int RangeEnd() const {
    if (range_end_ >= 0) {
      return std::max(range_end_, RangeStart() + 1);
    }
    return std::max({total_ - 1, end_ - 1, RangeStart() + 1});
  }
  void ToggleSimulation() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      sim_paused_ = !sim_paused_;
    }
    changed_.notify_all();
  }

  LiveViewerOptions options_;
  cbr::Ref<cbr::device::Device> device_;
  cbr::Ref<cbr::Scene> scene_;
  std::unique_ptr<cbr::app::Viewer> viewer_;
  std::vector<Surface> surfaces_;
  std::vector<FieldState> fields_;
  size_t frame_bytes_       = 0;
  float* d_values_          = nullptr;  // Colorize input
  float* d_ranges_          = nullptr;  // [min, max] per field
  cudaStream_t copy_stream_ = nullptr;
  cudaEvent_t publish_done_ = nullptr;
  // Rest bounding box
  std::array<double, 3> lo_ = {std::numeric_limits<double>::max(),
                               std::numeric_limits<double>::max(),
                               std::numeric_limits<double>::max()};
  std::array<double, 3> hi_ = {std::numeric_limits<double>::lowest(),
                               std::numeric_limits<double>::lowest(),
                               std::numeric_limits<double>::lowest()};
  uint64_t status_id_       = 0;
  uint64_t colorbar_id_     = 0;

  // Shared by the drawing thread and the producing thread
  std::mutex mutex_;
  std::condition_variable changed_;
  FrameHistory history_;
  std::vector<float> step_ms_;  // Per frame
  std::atomic<bool> stop_{false};
  std::atomic<bool> sim_paused_{false};
  std::atomic<bool> recording_done_{false};
  std::exception_ptr error_;

  // Timeline (drawing thread)
  Mode mode_        = Mode::Simulation;
  int total_        = 0;  // Expected, 0 if unknown
  int first_        = 0;  // Oldest kept
  int end_          = 0;  // Past the newest
  int cursor_       = -1;
  int shown_        = -1;
  bool follow_      = true;
  bool playing_     = false;
  bool loop_        = false;
  double fps_       = 30.0;
  bool every_frame_ = false;  // Else real time
  int range_start_  = 0;
  int range_end_    = -1;  // -1 follows the end
  int field_        = 0;
  // Shown in the colorbar
  int colorbar_field_                  = -1;
  std::array<float, 2> colorbar_range_ = {0.0f, 0.0f};
  Clock::time_point next_advance_;
};

void LiveViewer::Impl::AddSurface(const std::string& name,
                                  const LinearSurface& linear,
                                  const double* d_x12, const double* d_y12,
                                  const double* d_z12) {
  if (viewer_) {
    throw std::runtime_error("LiveViewer surfaces must be added before Run");
  }
  const int n_vertices = int(linear.rest.size());
  const size_t n_terms = size_t(n_vertices) * linear.terms;
  if (n_vertices == 0 || linear.terms < 1 || linear.coefs.size() != n_terms ||
      linear.weights.size() != n_terms) {
    throw std::runtime_error(
        "LiveViewer surface " + name +
        " needs terms coefficients and weights per vertex");
  }
  const bool has_elements = !linear.element_offsets.empty();
  if (has_elements &&
      (linear.element_offsets.size() != size_t(n_vertices) + 1 ||
       linear.element_offsets.front() != 0 ||
       size_t(linear.element_offsets.back()) != linear.elements.size())) {
    throw std::runtime_error("LiveViewer surface " + name +
                             " needs element offsets for every vertex");
  }

  std::vector<cbr::Float3> points(n_vertices);
  for (int i = 0; i < n_vertices; ++i) {
    const Eigen::Vector3d& v = linear.rest[i];
    points[i]                = {float(v.x()), float(v.y()), float(v.z())};
    for (int a = 0; a < 3; ++a) {
      lo_[a] = std::min(lo_[a], v[a]);
      hi_[a] = std::max(hi_[a], v[a]);
    }
  }

  cbr::Mesh::Desc desc;
  desc.points = points;
  desc.faceVertexCounts.assign(linear.triangles.size(), 3);
  desc.faceVertexIndices.reserve(linear.triangles.size() * 3);
  for (const Eigen::Vector3i& t : linear.triangles) {
    desc.faceVertexIndices.insert(desc.faceVertexIndices.end(),
                                  {t.x(), t.y(), t.z()});
  }
  // Loop shades smooth, with normals the renderer recomputes as points move
  desc.subdivisionScheme = cbr::SubdivisionScheme::Loop;
  const std::vector<cbr::Float3> colors(n_vertices,
                                        cbr::Float3{0.19f, 0.07f, 0.23f});
  desc.primvars = {cbr::Primvar::Create<cbr::Float3>(
      "displayColor", cbr::Interpolation::Vertex,
      cbr::Span<const cbr::Float3>(colors.data(), colors.size()))};

  Surface surface;
  surface.name       = name;
  surface.path       = "/" + name;
  surface.n_vertices = n_vertices;
  surface.terms      = linear.terms;
  surface.d_x12      = d_x12;
  surface.d_y12      = d_y12;
  surface.d_z12      = d_z12;
  scene_->Define<cbr::Mesh>(surface.path, desc);

  const size_t vec_bytes = size_t(n_vertices) * 3 * sizeof(float);
  HANDLE_ERROR(cudaMalloc(&surface.d_coefs, n_terms * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(surface.d_coefs, linear.coefs.data(),
                          n_terms * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc(&surface.d_weights, n_terms * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(surface.d_weights, linear.weights.data(),
                          n_terms * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc(&surface.d_rest, vec_bytes));
  HANDLE_ERROR(cudaMemcpy(surface.d_rest, points.data(), vec_bytes,
                          cudaMemcpyHostToDevice));
  if (has_elements) {
    const size_t offsets = linear.element_offsets.size() * sizeof(int);
    const size_t ids =
        std::max<size_t>(linear.elements.size(), 1) * sizeof(int);
    HANDLE_ERROR(cudaMalloc(&surface.d_element_offsets, offsets));
    HANDLE_ERROR(cudaMemcpy(surface.d_element_offsets,
                            linear.element_offsets.data(), offsets,
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMalloc(&surface.d_elements, ids));
    HANDLE_ERROR(cudaMemcpy(surface.d_elements, linear.elements.data(),
                            linear.elements.size() * sizeof(int),
                            cudaMemcpyHostToDevice));
  }
  surfaces_.push_back(surface);
}

// Ground, lights and camera framed on the surfaces, then the viewer
void LiveViewer::Impl::Open() {
  const double size = Size();
  if (fields_.empty()) {
    fields_.push_back({{"Displacement", "m", Displacement()}});
  }
  // Each surface's part of a frame: its points, then its values of every
  // field in turn
  frame_bytes_ = 0;
  for (Surface& s : surfaces_) {
    s.points_offset = frame_bytes_;
    s.values_offset =
        s.points_offset + size_t(s.n_vertices) * 3 * sizeof(float);
    frame_bytes_ =
        s.values_offset + fields_.size() * s.n_vertices * sizeof(float);
  }
  for (const FieldState& f : fields_) {
    if (f.field.source.kind == FieldSource::Kind::ElementScalar) {
      for (const Surface& s : surfaces_) {
        if (s.d_element_offsets == nullptr) {
          throw std::runtime_error("LiveViewer field " + f.field.name +
                                   " is per element, but " + "surface " +
                                   s.name + " doesn't know its elements");
        }
      }
    }
  }
  field_ = 0;
  if (!options_.field.empty()) {
    const auto named = std::find_if(
        fields_.begin(), fields_.end(),
        [&](const FieldState& f) { return f.field.name == options_.field; });
    if (named == fields_.end()) {
      std::string names;
      for (const FieldState& f : fields_) {
        names += (names.empty() ? "" : ", ") + f.field.name;
      }
      throw std::runtime_error("LiveViewer has no field " + options_.field +
                               " (fields are " + names + ")");
    }
    field_ = int(named - fields_.begin());
  }
  const cbr::Double3 center = {0.5 * (lo_[0] + hi_[0]), 0.5 * (lo_[1] + hi_[1]),
                               0.5 * (lo_[2] + hi_[2])};

  cbr::Plane::Desc ground;
  ground.translate     = {center.x, center.y, lo_[2] - 0.002 * size};
  ground.width         = float(8 * size);
  ground.length        = float(8 * size);
  ground.checkerboard  = true;
  ground.checkerSize   = float(size / 4);
  ground.checkerColorA = {0.55f, 0.55f, 0.58f};
  ground.checkerColorB = {0.35f, 0.35f, 0.38f};
  ground.visibility    = options_.show_ground ? cbr::Visibility::Inherited
                                              : cbr::Visibility::Invisible;
  scene_->Define<cbr::Plane>("/Ground", ground);

  cbr::DistantLight::Desc sun;
  sun.orient    = cbr::FromAzimuthElevation(135, 50);
  sun.intensity = 50000;
  scene_->Define<cbr::DistantLight>("/Sun", sun);

  cbr::DomeLight::Desc sky;
  sky.intensity = 4000;
  sky.color     = {0.75f, 0.85f, 1.0f};
  scene_->Define<cbr::DomeLight>("/Sky", sky);

  const cbr::Double3 eye = {center.x + 1.6 * size, center.y - 2.0 * size,
                            center.z + 0.9 * size};
  cbr::Camera::Desc camera;
  camera.translate   = eye;
  camera.orient      = cbr::LookAt(eye, center);
  camera.focalLength = 30;
  scene_->Define<cbr::Camera>("/Camera", camera);

  // Minimal shading is noise free on every frame, which suits a scene that
  // changes every frame better than the accumulating real-time mode
  scene_->Define<cbr::Scope>("/Render");
  scene_->Define<cbr::Scope>("/Render/Vars");
  cbr::RenderVar::Desc color;
  color.sourceName = "LdrColor";
  scene_->Define<cbr::RenderVar>("/Render/Vars/LdrColor", color);
  cbr::RenderProduct::Desc product;
  product.renderMode            = cbr::RenderMode::Minimal;
  product.mode                  = cbr::MinimalMode::TextureDiffuse;
  product.castShadows           = true;
  product.shadowVisibility      = 0.35f;
  product.ambientHemispheric    = true;
  product.ambientLightColor     = {0.75f, 0.85f, 1.0f};
  product.ambientGroundColor    = {0.35f, 0.33f, 0.3f};
  product.ambientLightIntensity = 0.35f;
  product.backgroundColor       = {0.62f, 0.72f, 0.85f};
  product.camera                = {"/Camera"};
  product.resolution            = {options_.width, options_.height};
  product.orderedVars           = {"/Render/Vars/LdrColor"};
  scene_->Define<cbr::RenderProduct>("/Render/Live", product);

  cbr::app::ViewerDesc desc;
  desc.product   = "/Render/Live";
  desc.title     = options_.title;
  desc.width     = options_.width;
  desc.height    = options_.height;
  desc.offscreen = options_.offscreen;
  viewer_        = std::make_unique<cbr::app::Viewer>(scene_, desc);

  cbr::app::ViewportDesc viewport;
  viewport.name     = kViewport;
  viewport.product  = "/Render/Live";
  viewport.replaces = "Viewport";
  viewport.footer   = [this] { DrawTimeline(); };
  viewer_->AddViewport(std::move(viewport));

  // Panels start hidden either way, a click away in the Window menu
  for (const char* panel : {"Stage", "Properties", "Info", "Log", "Profiler",
                            "Memory", "Capture"}) {
    try {
      viewer_->ShowPanel(panel, false);
    } catch (const cbr::Error&) {
      // Not in this build of the viewer
    }
  }
  viewer_->ShowBaseGui(options_.show_viewer_ui);

  viewer_->AddKeyHandler(
      [this](const cbr::app::KeyEvent& event) { return OnKey(event); });

  cbr::app::ColorbarDesc colorbar;
  colorbar.colormap = cbr::Colormap::Turbo;
  colorbar_id_      = viewer_->AddColorbar(colorbar, kViewport);
  UpdateColorbar();
  status_id_ = viewer_->AddText("", {}, kViewport);

  history_.Configure(frame_bytes_, options_.history_device_mb << 20,
                     options_.history_host_mb << 20);
  size_t largest = 0;
  for (const Surface& s : surfaces_) {
    largest = std::max(largest, size_t(s.n_vertices));
  }
  HANDLE_ERROR(cudaMalloc(&d_values_, largest * sizeof(float)));
  const std::vector<float> ranges = [&] {
    std::vector<float> r;
    for (size_t f = 0; f < fields_.size(); ++f) {
      r.insert(r.end(), {FLT_MAX, -FLT_MAX});
    }
    return r;
  }();
  HANDLE_ERROR(cudaMalloc(&d_ranges_, ranges.size() * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_ranges_, ranges.data(),
                          ranges.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
}

// The colormap's bottom and top for a field: its own range, or else one that
// fits the values recorded so far, symmetric about zero for signed values
std::array<float, 2> LiveViewer::Impl::ColorRange(int field) {
  const FieldState& f = fields_[field];
  float lo = f.field.min, hi = f.field.max;
  if (hi <= lo) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      lo = f.lo;
      hi = f.hi;
    }
    if (lo > hi) {
      lo = hi = 0.0f;  // Nothing recorded yet
    } else if (lo < 0.0f && hi > 0.0f) {
      hi = std::max(-lo, hi);
      lo = -hi;
    } else {
      lo = std::min(lo, 0.0f);
      hi = std::max(hi, 0.0f);
    }
    if (hi <= lo) {
      hi = lo + 1.0f;  // All zero
    }
  }
  return {lo * f.scale, hi * f.scale};
}

// Show the field on screen and its range in the colorbar, recoloring the
// frame on screen when either changed
void LiveViewer::Impl::UpdateColorbar() {
  const std::array<float, 2> range = ColorRange(field_);
  if (field_ == colorbar_field_ && range == colorbar_range_) {
    return;
  }
  const Field& f = fields_[field_].field;
  cbr::app::ColorbarDesc colorbar;
  colorbar.colormap = cbr::Colormap::Turbo;
  colorbar.minimum  = range[0];
  colorbar.maximum  = range[1];
  colorbar.label    = f.name;
  colorbar.units    = f.units;
  viewer_->SetColorbar(colorbar_id_, colorbar);
  colorbar_field_ = field_;
  colorbar_range_ = range;
  shown_          = -1;
}

bool LiveViewer::Impl::OnKey(const cbr::app::KeyEvent& event) {
  const bool press = event.action == cbr::app::InputAction::Press;
  const bool held  = press || event.action == cbr::app::InputAction::Repeat;
  if (press && event.key == "Space") {
    TogglePlay();
  } else if (held && event.key == "LeftArrow") {
    StepFrames(-1);
  } else if (held && event.key == "RightArrow") {
    StepFrames(1);
  } else if (press && event.key == "Home") {
    Seek(std::max(RangeStart(), first_));
  } else if (press && event.key == "End") {
    FollowNewest();
  } else if (press && event.key == "L") {
    loop_ = !loop_;
  } else if (press && event.key == "P" && mode_ == Mode::Simulation) {
    ToggleSimulation();
  } else {
    return false;
  }
  return true;
}

void LiveViewer::Impl::TogglePlay() {
  if (follow_) {
    // Stop following, holding the frame on screen
    follow_  = false;
    playing_ = false;
    return;
  }
  playing_ = !playing_;
  if (playing_) {
    const int last = std::min(RangeEnd(), end_ - 1);
    if (cursor_ >= last && !loop_) {
      cursor_ = std::max(RangeStart(), first_);
    }
    next_advance_ = Clock::now();
  }
}

// Producing thread. Gather the surface nodes into a new frame, after the step
// that just finished on the legacy default stream.
void LiveViewer::Impl::Record(double step_ms) {
  char* slot = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    slot = static_cast<char*>(history_.Acquire());
  }
  for (const FieldState& f : fields_) {
    if (f.field.update) {
      f.field.update();
    }
  }
  for (const Surface& s : surfaces_) {
    float* points = reinterpret_cast<float*>(slot + s.points_offset);
    GatherLivePoints(s.n_vertices, s.terms, s.d_coefs, s.d_weights, s.d_x12,
                     s.d_y12, s.d_z12, points, nullptr);
    for (size_t f = 0; f < fields_.size(); ++f) {
      float* values = reinterpret_cast<float*>(
          slot + s.values_offset + f * s.n_vertices * sizeof(float));
      RecordField(fields_[f].field.source, s, points, values);
      WidenLiveRange(s.n_vertices, values, d_ranges_ + 2 * f, nullptr);
    }
  }
  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaStreamSynchronize(nullptr));
  std::vector<float> ranges(2 * fields_.size());
  HANDLE_ERROR(cudaMemcpy(ranges.data(), d_ranges_,
                          ranges.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
  std::lock_guard<std::mutex> lock(mutex_);
  history_.Commit(slot);
  step_ms_.push_back(float(step_ms));
  for (size_t f = 0; f < fields_.size(); ++f) {
    fields_[f].lo = ranges[2 * f];
    fields_[f].hi = ranges[2 * f + 1];
  }
}

// Producing thread. One field's values on one surface of a new frame.
void LiveViewer::Impl::RecordField(const FieldSource& source, const Surface& s,
                                   const float* points, float* values) {
  const int component = int(source.component);
  switch (source.kind) {
    case FieldSource::Kind::Displacement:
      LiveDisplacement(s.n_vertices, points, s.d_rest, component, values,
                       nullptr);
      break;
    case FieldSource::Kind::NodeScalar:
      LiveNodeScalar(s.n_vertices, s.terms, s.d_coefs, source.x, values,
                     nullptr);
      break;
    case FieldSource::Kind::NodeVector:
      LiveNodeVector(s.n_vertices, s.terms, s.d_coefs, s.d_weights, source.x,
                     source.y, source.z, source.stride, component, values,
                     nullptr);
      break;
    case FieldSource::Kind::ElementScalar:
      LiveElementScalar(s.n_vertices, s.d_element_offsets, s.d_elements,
                        source.x, values, nullptr);
      break;
    case FieldSource::Kind::Kernel: {
      FieldContext ctx;
      ctx.surface    = s.name;
      ctx.n_vertices = s.n_vertices;
      ctx.terms      = s.terms;
      ctx.coefs      = s.d_coefs;
      ctx.weights    = s.d_weights;
      ctx.points     = points;
      ctx.rest       = s.d_rest;
      source.kernel(ctx, values, nullptr);
      break;
    }
  }
}

// Drawing thread. Copy a recorded frame into the renderer's own buffers,
// coloring it on the way. Queues the work without waiting for it (the
// renderer waits on the GPU); returns false, doing nothing, while the
// previous publish is still in flight.
bool LiveViewer::Impl::Publish(int frame) {
  std::lock_guard<std::mutex> lock(mutex_);
  const cudaError_t previous = cudaEventQuery(publish_done_);
  if (previous == cudaErrorNotReady) {
    return false;
  }
  HANDLE_ERROR(previous);
  const char* data                 = history_.Data(frame);
  const std::array<float, 2> range = colorbar_range_;
  for (const Surface& s : surfaces_) {
    cbr::MeshView view  = scene_->Select<cbr::Mesh>(s.path);
    cbr::Mapping points = view.Points().Map(cbr::MapAccess::Write);
    cbr::Mapping colors =
        view.Primvars().MapPrimvar("displayColor", cbr::MapAccess::Write);
    // The frame may be in device or pinned host memory
    HANDLE_ERROR(cudaMemcpyAsync(MappedFloat3(points, s.n_vertices, "points"),
                                 data + s.points_offset,
                                 size_t(s.n_vertices) * 3 * sizeof(float),
                                 cudaMemcpyDefault, copy_stream_));
    const size_t field_bytes = size_t(s.n_vertices) * sizeof(float);
    HANDLE_ERROR(cudaMemcpyAsync(d_values_,
                                 data + s.values_offset + field_ * field_bytes,
                                 field_bytes, cudaMemcpyDefault, copy_stream_));
    ColorizeLiveSurface(s.n_vertices, d_values_, range[0], range[1],
                        MappedFloat3(colors, s.n_vertices, "displayColor"),
                        copy_stream_);
    HANDLE_ERROR(cudaGetLastError());
    const cbr::Stream producer = cbr::Stream::Compute(copy_stream_);
    colors.Close(producer);
    points.Close(producer);
  }
  // The frame's slot can't be moved or reused until the copies finished
  HANDLE_ERROR(cudaEventRecord(publish_done_, copy_stream_));
  history_.SetReader(frame, publish_done_);
  return true;
}

void LiveViewer::Impl::UpdateTimeline() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    first_ = history_.First();
    end_   = history_.End();
  }
  if (end_ == 0) {
    return;
  }
  const auto now = Clock::now();
  if (follow_) {
    cursor_ = end_ - 1;
  } else if (playing_ && now >= next_advance_) {
    const int low     = std::max(RangeStart(), first_);
    const int high    = std::min(RangeEnd(), end_ - 1);
    const auto period = std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(1.0 / fps_));
    // Real time skips the frames that are due; every frame shows each one
    const int due = every_frame_ ? 1 : 1 + int((now - next_advance_) / period);
    for (int i = 0; i < due; ++i) {
      if (cursor_ < high) {
        ++cursor_;
      } else if (loop_) {
        cursor_ = low;
      } else {
        // The end of the range, or the newest frame while more are coming
        if (recording_done_ || high < end_ - 1 || range_end_ >= 0) {
          playing_ = false;
        }
        break;
      }
    }
    next_advance_ = every_frame_ ? std::max(next_advance_ + period, now)
                                 : next_advance_ + due * period;
  }
  cursor_ = std::clamp(cursor_, first_, end_ - 1);
}

void LiveViewer::Impl::Drive(std::thread& producer) {
  const auto start = Clock::now();
  int frames       = 0;
  int drawn_at_end = 0;
  while (viewer_->Open()) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (error_) {
        break;
      }
    }
    UpdateTimeline();
    UpdateColorbar();
    if (cursor_ >= first_ && cursor_ < end_ && cursor_ != shown_) {
      if (Publish(cursor_)) {
        shown_ = cursor_;
      }
    }

    float last_ms = 0.0f;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      last_ms = step_ms_.empty() ? 0.0f : step_ms_.back();
    }
    char status[192];
    if (mode_ == Mode::Simulation) {
      const std::string of =
          total_ > 0 ? " of " + std::to_string(total_) : std::string();
      std::snprintf(status, sizeof(status), "%d steps%s   %.1f ms/step%s", end_,
                    of.c_str(), last_ms,
                    sim_paused_       ? "   simulation paused (P)"
                    : recording_done_ ? "   done"
                                      : "");
    } else {
      std::snprintf(status, sizeof(status), "%d of %d frames loaded", end_,
                    total_);
    }
    viewer_->SetText(status_id_, status);

    viewer_->Frame();
    ++frames;
    // Offscreen runs end once the last frame is drawn (rendering lags a few
    // frames behind the update)
    const bool at_end = recording_done_ && shown_ == end_ - 1 &&
                        (mode_ == Mode::Simulation || (!playing_ && !loop_));
    if (options_.offscreen && at_end && ++drawn_at_end > 3) {
      break;
    }
  }

  stop_ = true;
  changed_.notify_all();
  producer.join();
  if (error_) {
    std::rethrow_exception(error_);
  }
  const double seconds =
      std::chrono::duration<double>(Clock::now() - start).count();
  std::printf("live view: %d frames recorded, %d drawn in %.1f s (%.1f fps)\n",
              end_, frames, seconds, frames / seconds);
  if (!options_.screenshot.empty()) {
    SavePng(options_.screenshot);
    std::printf("wrote %s\n", options_.screenshot.c_str());
  }
}

void LiveViewer::Impl::Run(int n_steps, const std::function<void(int)>& step) {
  if (!viewer_) {
    Open();
  }
  mode_    = Mode::Simulation;
  total_   = std::max(n_steps, 0);
  follow_  = true;
  playing_ = false;

  std::thread producer([&] {
    try {
      for (int i = 0; (n_steps <= 0 || i < n_steps) && !stop_; ++i) {
        {
          std::unique_lock<std::mutex> lock(mutex_);
          changed_.wait(lock, [&] { return !sim_paused_ || stop_; });
        }
        if (stop_) {
          break;
        }
        const auto t0 = Clock::now();
        step(i);
        Record(std::chrono::duration<double, std::milli>(Clock::now() - t0)
                   .count());
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      error_ = std::current_exception();
    }
    recording_done_ = true;
  });
  Drive(producer);
}

void LiveViewer::Impl::Play(int n_frames, double fps, bool loop,
                            const std::function<void(int)>& load) {
  if (!viewer_) {
    Open();
  }
  if (n_frames <= 0) {
    throw std::runtime_error("LiveViewer has no frames to play");
  }
  mode_         = Mode::Playback;
  total_        = n_frames;
  follow_       = false;
  playing_      = true;
  loop_         = loop;
  fps_          = std::max(fps, 1e-3);
  cursor_       = 0;
  next_advance_ = Clock::now();

  std::thread producer([&] {
    try {
      for (int i = 0; i < n_frames && !stop_; ++i) {
        const auto t0 = Clock::now();
        load(i);
        Record(std::chrono::duration<double, std::milli>(Clock::now() - t0)
                   .count());
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      error_ = std::current_exception();
    }
    recording_done_ = true;
  });
  Drive(producer);
}

// The timeline strip under the view, laid out like Kit's timeline panel: the
// range's start, the ruler, its end and the frame on screen, then
// the transport and its settings
void LiveViewer::Impl::DrawTimeline() {
  const float unit    = ImGui::GetFontSize();
  const float field   = 4.0f * unit;
  const float spacing = ImGui::GetStyle().ItemSpacing.x;
  float icon          = 0.0f;
  for (const char* glyph : {kIconSkipBack, kIconStepBack, kIconPlay,
                            kIconSquare, kIconRepeat, kIconSettings}) {
    icon = std::max(icon, ImGui::CalcTextSize(glyph).x);
  }
  const float button = icon + 2.0f * ImGui::GetStyle().FramePadding.x;

  const auto frame_field = [&](const char* id, int value, const char* tip) {
    ImGui::SetNextItemWidth(field);
    ImGui::InputInt(id, &value, 0, 0);
    ImGui::SetItemTooltip("%s", tip);
    return ImGui::IsItemDeactivatedAfterEdit() ? value : -1;
  };
  if (const int start = frame_field("##start", RangeStart(), "Start frame");
      start >= 0) {
    range_start_ = std::min(start, RangeEnd() - 1);
  }
  ImGui::SameLine();
  const float picker = fields_.size() > 1 ? 9.0f * unit + spacing : 0.0f;
  const float right  = 2.0f * field + 8.0f * button + 10.0f * spacing + picker;
  DrawRuler(std::max(ImGui::GetContentRegionAvail().x - right, 4.0f * unit));
  ImGui::SameLine();
  if (const int end = frame_field("##end", RangeEnd(), "End frame"); end >= 0) {
    range_end_ = std::max(end, RangeStart() + 1);
  }
  ImGui::SameLine();
  if (const int now =
          frame_field("##now", std::max(cursor_, 0), "Current frame");
      now >= 0) {
    Seek(now);
  }
  ImGui::SameLine();

  const auto transport = [](const char* glyph, const char* id, const char* tip,
                            bool lit) {
    if (lit) {
      ImGui::PushStyleColor(ImGuiCol_Button,
                            ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
    }
    const bool clicked = ImGui::Button((std::string(glyph) + id).c_str());
    if (lit) {
      ImGui::PopStyleColor();
    }
    ImGui::SetItemTooltip("%s", tip);
    ImGui::SameLine();
    return clicked;
  };
  if (transport(kIconSkipBack, "##first", "First frame (Home)", false)) {
    Seek(std::max(RangeStart(), first_));
  }
  if (transport(kIconStepBack, "##back", "Previous frame (Left)", false)) {
    StepFrames(-1);
  }
  if (transport(playing_ ? kIconPause : kIconPlay, "##play",
                playing_ ? "Pause (Space)" : "Play (Space)", false)) {
    TogglePlay();
  }
  if (transport(kIconSquare, "##stop", "Stop: back to the start", false)) {
    Stop();
  }
  if (transport(kIconStepForward, "##forward", "Next frame (Right)", false)) {
    StepFrames(1);
  }
  if (transport(kIconSkipForward, "##last",
                follow_ ? "Last frame: following new frames (End)"
                        : "Last frame, then follow new frames (End)",
                follow_)) {
    FollowNewest();
  }
  if (transport(kIconRepeat, "##loop",
                loop_ ? "Looping: on (L)" : "Looping: off (L)", loop_)) {
    loop_ = !loop_;
  }
  if (fields_.size() > 1) {
    ImGui::SetNextItemWidth(9.0f * unit);
    if (ImGui::BeginCombo("##field", fields_[field_].field.name.c_str())) {
      for (int f = 0; f < int(fields_.size()); ++f) {
        if (ImGui::Selectable(fields_[f].field.name.c_str(), f == field_)) {
          SelectField(f);
        }
      }
      ImGui::EndCombo();
    }
    ImGui::SetItemTooltip("Field the surfaces are colored by");
    ImGui::SameLine();
  }
  if (ImGui::Button((std::string(kIconSettings) + "##settings").c_str())) {
    ImGui::OpenPopup("##timelineSettings");
  }
  ImGui::SetItemTooltip("Timeline settings");
  ImGui::SetNextWindowSizeConstraints(ImVec2(16.0f * unit, 0.0f),
                                      ImVec2(FLT_MAX, FLT_MAX));
  if (ImGui::BeginPopup("##timelineSettings")) {
    DrawTimelineSettings();
    ImGui::EndPopup();
  }
}

// The ruler across the range: ticks labeled at least five characters apart,
// the recorded frames shaded, the frame on screen's line, and a click or drag
// that pauses and scrubs
void LiveViewer::Impl::DrawRuler(float width) {
  const double start = RangeStart();
  const double end   = RangeEnd();
  const float height = ImGui::GetFrameHeight();
  const ImVec2 low   = ImGui::GetCursorScreenPos();
  const ImVec2 high(low.x + width, low.y + height);
  ImGui::InvisibleButton("##ruler", ImVec2(width, height));
  const double per_frame = double(width) / std::max(end - start, 1e-9);
  const auto x_at        = [&](double f) {
    return low.x + float((std::clamp(f, start, end) - start) * per_frame);
  };
  if (ImGui::IsItemActive() && end_ > 0) {
    const double at =
        start + double(ImGui::GetIO().MousePos.x - low.x) / per_frame;
    Seek(int(std::clamp(std::round(at), start, end)));
  }

  ImDrawList* draw    = ImGui::GetWindowDrawList();
  const float round   = ImGui::GetStyle().FrameRounding;
  const ImVec4 accent = ImGui::GetStyleColorVec4(ImGuiCol_SeparatorActive);
  draw->AddRectFilled(low, high, ImGui::GetColorU32(ImGuiCol_FrameBg), round);
  if (end_ > 0 && end_ - 1 >= start && first_ <= end) {
    ImVec4 recorded = accent;
    recorded.w      = 0.25f;
    draw->AddRectFilled(
        ImVec2(x_at(first_), low.y),
        ImVec2(std::max(x_at(end_ - 1), x_at(first_) + 2.0f), high.y),
        ImGui::GetColorU32(recorded), round);
    if (first_ > start) {
      draw->AddRectFilled(low, ImVec2(x_at(first_), high.y),
                          ImGui::GetColorU32(ImGuiCol_TextDisabled, 0.25f),
                          round);
    }
  }

  // Labeled ticks at 1, 2 or 5 times a power of ten frames, and unlabeled
  // fifths between them where they fit
  const double least = (ImGui::CalcTextSize("00000").x + 8.0f) / per_frame;
  const double decade =
      std::pow(10.0, std::floor(std::log10(std::max(least, 1.0))));
  double major = 10.0 * decade;
  for (const double factor : {1.0, 2.0, 5.0}) {
    if (factor * decade >= least) {
      major = factor * decade;
      break;
    }
  }
  const double minor = std::max(major / 5.0, 1.0);
  const bool minors  = minor * per_frame >= 5.0;
  const ImU32 tick   = ImGui::GetColorU32(ImGuiCol_TextDisabled);
  for (double t = std::ceil(start / minor) * minor; t <= end; t += minor) {
    const float x      = x_at(t);
    const bool labeled = std::fabs(t / major - std::round(t / major)) < 1e-6;
    if (!labeled && !minors) {
      continue;
    }
    draw->AddLine(ImVec2(x, high.y - (labeled ? 0.45f : 0.2f) * height),
                  ImVec2(x, high.y), tick);
    if (labeled) {
      char label[32];
      std::snprintf(label, sizeof(label), "%g", t);
      draw->AddText(ImVec2(x + 2.0f, low.y), tick, label);
    }
  }

  if (cursor_ >= start && cursor_ <= end) {
    const float x    = x_at(cursor_);
    const ImU32 line = ImGui::GetColorU32(accent);
    draw->AddLine(ImVec2(x, low.y), ImVec2(x, high.y), line, 2.0f);
    const float half = 0.2f * height;
    draw->AddTriangleFilled(ImVec2(x - half, low.y), ImVec2(x + half, low.y),
                            ImVec2(x, low.y + half), line);
  }
  if (ImGui::IsItemHovered() && !ImGui::IsItemActive()) {
    ImGui::SetTooltip("Click or drag to scrub");
  }
}

// The frame rate and playback settings, then the live view's own
void LiveViewer::Impl::DrawTimelineSettings() {
  const auto section = [](const char* title) {
    ImGui::TextDisabled("%s", title);
  };
  section("Frame Rate");
  for (const double fps : {24.0, 25.0, 30.0, 48.0, 50.0, 60.0, 120.0}) {
    char label[32];
    std::snprintf(label, sizeof(label), "%g fps", fps);
    if (ImGui::MenuItem(label, nullptr, fps_ == fps)) {
      fps_ = fps;
    }
  }
  ImGui::Separator();
  section("Playback");
  if (ImGui::MenuItem("Play Every Frame", nullptr, every_frame_)) {
    every_frame_ = !every_frame_;
  }
  ImGui::SetItemTooltip(
      "A frame each step however long it takes, rather than real time");
  if (ImGui::MenuItem("Loop", "L", loop_)) {
    loop_ = !loop_;
  }
  if (ImGui::MenuItem("Follow New Frames", "End", follow_)) {
    FollowNewest();
  }
  if (ImGui::MenuItem("Whole Range", nullptr,
                      range_start_ == 0 && range_end_ < 0)) {
    range_start_ = 0;
    range_end_   = -1;
  }
  ImGui::SetItemTooltip("The range from the first frame to the last");
  if (mode_ == Mode::Simulation) {
    ImGui::Separator();
    section("Simulation");
    if (ImGui::MenuItem("Pause Simulation", "P", bool(sim_paused_))) {
      ToggleSimulation();
    }
  }
  ImGui::Separator();
  section("View");
  if (ImGui::MenuItem("Show Viewer UI", nullptr, options_.show_viewer_ui)) {
    options_.show_viewer_ui = !options_.show_viewer_ui;
    viewer_->ShowBaseGui(options_.show_viewer_ui);
  }
  ImGui::SetItemTooltip(
      "The viewer's menus, toolbar, HUD, axis indicator and panels");
  if (ImGui::MenuItem("Show Ground", nullptr, options_.show_ground)) {
    options_.show_ground = !options_.show_ground;
    cbr::PlanePrim(*scene_->Find("/Ground"))
        .SetVisibility(options_.show_ground ? cbr::Visibility::Inherited
                                            : cbr::Visibility::Invisible);
  }
  ImGui::Separator();
  section("Color");
  ImGui::SetNextItemWidth(-FLT_MIN);
  ImGui::SliderFloat("##scale", &fields_[field_].scale, 0.05f, 4.0f,
                     "range x%.2f", ImGuiSliderFlags_Logarithmic);
  ImGui::SetItemTooltip("Scale of the colormap's range, for this field");
  for (int f = 0; f < int(fields_.size()); ++f) {
    if (ImGui::MenuItem(fields_[f].field.name.c_str(), nullptr, f == field_)) {
      SelectField(f);
    }
  }

  ImGui::Separator();
  section("History");
  std::vector<float> step_ms;
  size_t device_bytes = 0, host_bytes = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    step_ms      = step_ms_;
    device_bytes = history_.DeviceBytes();
    host_bytes   = history_.HostBytes();
  }
  ImGui::Text("%d frames kept (%d dropped)", end_ - first_, first_);
  ImGui::Text("%.0f MB device, %.0f MB host", device_bytes / 1048576.0,
              host_bytes / 1048576.0);
  if (!step_ms.empty()) {
    const float most = *std::max_element(step_ms.begin(), step_ms.end());
    ImGui::PlotLines(
        "##step time", step_ms.data(), int(step_ms.size()), 0,
        mode_ == Mode::Simulation ? "step time (ms)" : "load time (ms)", 0.0f,
        most * 1.1f,
        ImVec2(16.0f * ImGui::GetFontSize(), 3.0f * ImGui::GetFontSize()));
  }
}

void LiveViewer::Impl::SavePng(const std::string& path) {
  if (!viewer_) {
    return;
  }
  const std::vector<uint8_t> rgba = viewer_->ReadPixels();
  const uint32_t w = options_.width, h = options_.height;
  if (rgba.size() != size_t(w) * h * 4) {
    throw std::runtime_error("LiveViewer read " + std::to_string(rgba.size()) +
                             " pixel bytes, expected " +
                             std::to_string(size_t(w) * h * 4));
  }
  cbr::WritePng(path, cbr::Pixels(rgba, h, w, 4));
}

LiveViewer::LiveViewer(const LiveViewerOptions& options)
    : impl_(std::make_unique<Impl>(options)) {}

LiveViewer::~LiveViewer() = default;

void LiveViewer::AddSurface(const std::string& name,
                            const LinearSurface& surface, const double* d_x12,
                            const double* d_y12, const double* d_z12) {
  impl_->AddSurface(name, surface, d_x12, d_y12, d_z12);
}

namespace {

// One term per vertex, its node's position
LinearSurface NodeSurface(const ANCFCPUUtils::SurfaceTriMesh& mesh,
                          const std::string& name) {
  if (mesh.global_node_ids.size() != mesh.vertices.size()) {
    throw std::runtime_error("LiveViewer surface " + name +
                             " needs one node ID per vertex");
  }
  LinearSurface linear;
  linear.terms     = 1;
  linear.coefs     = mesh.global_node_ids;
  linear.weights   = std::vector<float>(mesh.vertices.size(), 1.0f);
  linear.rest      = mesh.vertices;
  linear.triangles = mesh.triangles;
  return linear;
}

}  // namespace

void LiveViewer::AddSurface(const std::string& name,
                            const ANCFCPUUtils::SurfaceTriMesh& mesh,
                            const double* d_x12, const double* d_y12,
                            const double* d_z12) {
  impl_->AddSurface(name, NodeSurface(mesh, name), d_x12, d_y12, d_z12);
}

void LiveViewer::AddSurface(const std::string& name,
                            const Eigen::MatrixXd& nodes,
                            const Eigen::MatrixXi& elements,
                            const ANCFCPUUtils::MeshInstance& instance,
                            const double* d_x12, const double* d_y12,
                            const double* d_z12) {
  const ANCFCPUUtils::SurfaceTriMesh mesh =
      ANCFCPUUtils::ExtractSurfaceTriMesh(nodes, elements, instance);
  std::printf("%s surface: %zu vertices, %zu triangles\n", name.c_str(),
              mesh.vertices.size(), mesh.triangles.size());
  LinearSurface linear = NodeSurface(mesh, name);

  std::vector<std::vector<int>> node_elements(nodes.rows());
  for (int e = instance.element_offset;
       e < instance.element_offset + instance.num_elements; ++e) {
    for (int j = 0; j < elements.cols(); ++j) {
      node_elements.at(elements(e, j)).push_back(e);
    }
  }
  linear.element_offsets.push_back(0);
  for (const int node : mesh.global_node_ids) {
    const std::vector<int>& in = node_elements.at(node);
    linear.elements.insert(linear.elements.end(), in.begin(), in.end());
    linear.element_offsets.push_back(int(linear.elements.size()));
  }
  impl_->AddSurface(name, linear, d_x12, d_y12, d_z12);
}

void LiveViewer::AddField(Field field) {
  impl_->AddField(std::move(field));
}

void LiveViewer::Run(int n_steps, const std::function<void(int)>& step) {
  impl_->Run(n_steps, step);
}

void LiveViewer::Play(int n_frames, double fps, bool loop,
                      const std::function<void(int)>& load) {
  impl_->Play(n_frames, fps, loop, load);
}

namespace {

// A command line option and the LiveViewerOptions member it sets
struct ViewerFlag {
  const char* name;
  std::variant<bool LiveViewerOptions::*, int LiveViewerOptions::*,
               double LiveViewerOptions::*, std::string LiveViewerOptions::*>
      member;
  const char* help;
};

const ViewerFlag kViewerFlags[] = {
    {"offscreen", &LiveViewerOptions::offscreen, "render without a window"},
    {"ground", &LiveViewerOptions::show_ground, "show the ground plane"},
    {"viewer_ui", &LiveViewerOptions::show_viewer_ui,
     "show the viewer's menus, toolbar, HUD, axes and panels"},
    {"screenshot", &LiveViewerOptions::screenshot,
     "write the last frame to this PNG file (implies --offscreen)"},
    {"field", &LiveViewerOptions::field, "field to color by at first"},
};

}  // namespace

void AddViewerOptions(ANCFCPUUtils::Cli& cli, bool show_ground) {
  LiveViewerOptions defaults;
  defaults.show_ground = show_ground;
  for (const ViewerFlag& flag : kViewerFlags) {
    std::visit(
        [&](auto member) {
          const auto& value = defaults.*member;
          using T           = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, bool>) {
            cli.AddBool(flag.name, value, flag.help);
          } else if constexpr (std::is_same_v<T, int>) {
            cli.AddInt(flag.name, value, flag.help);
          } else if constexpr (std::is_same_v<T, double>) {
            cli.AddDouble(flag.name, value, flag.help);
          } else {
            cli.AddString(flag.name, value, flag.help);
          }
        },
        flag.member);
  }
}

LiveViewerOptions ViewerOptions(const ANCFCPUUtils::Cli& cli,
                                const std::string& title) {
  LiveViewerOptions options;
  options.title = title;
  for (const ViewerFlag& flag : kViewerFlags) {
    std::visit(
        [&](auto member) {
          auto& value = options.*member;
          using T     = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, bool>) {
            value = cli.GetBool(flag.name);
          } else if constexpr (std::is_same_v<T, int>) {
            value = cli.GetInt(flag.name);
          } else if constexpr (std::is_same_v<T, double>) {
            value = cli.GetDouble(flag.name);
          } else {
            value = cli.GetString(flag.name);
          }
        },
        flag.member);
  }
  return options;
}

FieldSource Displacement(Component component) {
  FieldSource source;
  source.kind      = FieldSource::Kind::Displacement;
  source.component = component;
  return source;
}

FieldSource NodeScalar(const double* d_values) {
  FieldSource source;
  source.kind = FieldSource::Kind::NodeScalar;
  source.x    = d_values;
  return source;
}

FieldSource NodeVector(const double* d_xyz, Component component) {
  FieldSource source;
  source.kind      = FieldSource::Kind::NodeVector;
  source.component = component;
  source.x         = d_xyz;
  source.y         = d_xyz + 1;
  source.z         = d_xyz + 2;
  source.stride    = 3;
  return source;
}

FieldSource NodeVector(const double* d_x, const double* d_y, const double* d_z,
                       Component component) {
  FieldSource source;
  source.kind      = FieldSource::Kind::NodeVector;
  source.component = component;
  source.x         = d_x;
  source.y         = d_y;
  source.z         = d_z;
  return source;
}

FieldSource ElementScalar(const double* d_values) {
  FieldSource source;
  source.kind = FieldSource::Kind::ElementScalar;
  source.x    = d_values;
  return source;
}

FieldSource Kernel(FieldKernel kernel) {
  FieldSource source;
  source.kind   = FieldSource::Kind::Kernel;
  source.kernel = std::move(kernel);
  return source;
}

}  // namespace LiveView
