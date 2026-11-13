/**
 * Chrono Batch Render Interop Test
 *
 * Creates a one-triangle Chrono Batch Render scene, writes its points from
 * CUDA through a renderer mapping, renders a few offscreen frames, and reports
 * what the mapping exposed (DLPack device type, aliasing). It is used as a
 * smoke test for the renderer integration and its CUDA interop.
 *
 * Example:
 *   bazel run //lib_bin/live_view:test_batch_render_interop -- vulkan
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "chrono_batch_render/app/viewer/viewer.h"
#include "chrono_batch_render/chrono_batch_render.h"

namespace cbr = chrono::batch_render;

int main(int argc, char** argv) {
  cbr::device::DeviceDesc device_desc;
  if (argc > 1 && std::strcmp(argv[1], "cuda") == 0) {
    device_desc.backend = cbr::device::Backend::Cuda;
  } else if (argc > 1 && std::strcmp(argv[1], "vulkan") == 0) {
    device_desc.backend = cbr::device::Backend::Vulkan;
  }
  device_desc.presentation = true;
  cbr::Ref<cbr::device::Device> device =
      cbr::device::Device::Create(device_desc);
  std::printf(
      "backend %s, adapter %s\n",
      std::string(cbr::device::BackendName(device->Adapter().backend)).c_str(),
      device->Adapter().name.c_str());

  cbr::Ref<cbr::Scene> scene = cbr::Scene::Create(*device);

  cbr::Mesh::Desc mesh;
  mesh.points            = {{-1, 0, 0}, {1, 0, 0}, {0, 0, 1.5f}};
  mesh.faceVertexCounts  = {3};
  mesh.faceVertexIndices = {0, 1, 2};
  mesh.subdivisionScheme = cbr::SubdivisionScheme::None;
  mesh.displayColor      = {0.8f, 0.3f, 0.2f};
  scene->Define<cbr::Mesh>("/Tri", mesh);

  cbr::DistantLight::Desc sun;
  sun.orient = cbr::FromAzimuthElevation(135, 45);
  scene->Define<cbr::DistantLight>("/Sun", sun);

  cbr::Camera::Desc camera;
  camera.translate = {0, -5, 1};
  camera.orient    = cbr::LookAt({0, -5, 1}, {0, 0, 0.5});
  scene->Define<cbr::Camera>("/Camera", camera);

  // Move the apex from CUDA, writing into the renderer's own memory
  {
    cbr::ArrayAttribute<cbr::Float3> points =
        scene->Select<cbr::Mesh>("/Tri").Points();
    cbr::Mapping mapping  = points.Map(cbr::MapAccess::Write);
    const DLTensor values = mapping.DLPack();
    std::printf(
        "mapping: device_type %d, aliased %d, ndim %d, bytes offset %llu\n",
        int(values.device.device_type), int(mapping.Aliased()), values.ndim,
        static_cast<unsigned long long>(values.byte_offset));
    const std::vector<float> moved = {-1, 0, 0, 1, 0, 0, 0, 0, 2};
    if (values.device.device_type == kDLCUDA) {
      void* dst       = static_cast<char*>(values.data) + values.byte_offset;
      cudaError_t err = cudaMemcpy(dst, moved.data(), moved.size() * 4,
                                   cudaMemcpyHostToDevice);
      std::printf("cudaMemcpy into mapping: %s\n", cudaGetErrorString(err));
    }
    mapping.Close(cbr::Stream::Default());
  }

  cbr::app::ViewerDesc viewer_desc;
  viewer_desc.offscreen = true;
  viewer_desc.width     = 320;
  viewer_desc.height    = 240;
  cbr::app::Viewer viewer(scene, viewer_desc);
  for (int i = 0; i < 3; ++i) {
    viewer.Frame();
  }
  const std::vector<uint8_t> pixels = viewer.ReadPixels();
  size_t lit                        = 0;
  for (uint8_t p : pixels) {
    lit += p > 8;
  }
  std::printf("read %zu bytes, %zu above black\n", pixels.size(), lit);
  return 0;
}
