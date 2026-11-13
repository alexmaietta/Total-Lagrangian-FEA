#include "LiveViewerKernels.cuh"

namespace LiveView {
namespace {

// Polynomial fit of Google's Turbo colormap (Mikhailov 2019), t in [0, 1].
__device__ float3 turbo_colormap(float t) {
  t              = fminf(fmaxf(t, 0.0f), 1.0f);
  const float t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
  float r = 0.13572138f + 4.61539260f * t - 42.66032258f * t2 +
            132.13108234f * t3 - 152.94239396f * t4 + 59.28637943f * t5;
  float g = 0.09140261f + 2.19418839f * t + 4.84296658f * t2 -
            14.18503333f * t3 + 4.27729857f * t4 + 2.82956604f * t5;
  float b = 0.10667330f + 12.64194608f * t - 60.58204836f * t2 +
            110.36276771f * t3 - 89.90310912f * t4 + 27.34824973f * t5;
  return make_float3(fminf(fmaxf(r, 0.0f), 1.0f), fminf(fmaxf(g, 0.0f), 1.0f),
                     fminf(fmaxf(b, 0.0f), 1.0f));
}

__global__ void gather_live_surface_kernel(
    int n_vertices, int terms, const int* d_coefs, const float* d_weights,
    const double* d_x12, const double* d_y12, const double* d_z12,
    const float* d_rest, const double* d_scalar, float* d_points,
    float* d_values) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_vertices) {
    return;
  }
  double px = 0.0, py = 0.0, pz = 0.0;
  for (int k = 0; k < terms; ++k) {
    const int c    = d_coefs[i * terms + k];
    const double w = d_weights[i * terms + k];
    px += w * d_x12[c];
    py += w * d_y12[c];
    pz += w * d_z12[c];
  }
  d_points[3 * i + 0] = static_cast<float>(px);
  d_points[3 * i + 1] = static_cast<float>(py);
  d_points[3 * i + 2] = static_cast<float>(pz);

  if (d_scalar != nullptr) {
    d_values[i] = static_cast<float>(d_scalar[d_coefs[i * terms]]);
  } else {
    const float dx = static_cast<float>(px) - d_rest[3 * i + 0];
    const float dy = static_cast<float>(py) - d_rest[3 * i + 1];
    const float dz = static_cast<float>(pz) - d_rest[3 * i + 2];
    d_values[i]    = sqrtf(dx * dx + dy * dy + dz * dz);
  }
}

__global__ void colorize_live_surface_kernel(int n_vertices,
                                             const float* d_values,
                                             float inv_max, float* d_colors) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_vertices) {
    return;
  }
  const float3 c      = turbo_colormap(d_values[i] * inv_max);
  d_colors[3 * i + 0] = c.x;
  d_colors[3 * i + 1] = c.y;
  d_colors[3 * i + 2] = c.z;
}

constexpr int kThreads = 256;

int Blocks(int n) {
  return (n + kThreads - 1) / kThreads;
}

}  // namespace

void GatherLiveSurface(int n_vertices, int terms, const int* d_coefs,
                       const float* d_weights, const double* d_x12,
                       const double* d_y12, const double* d_z12,
                       const float* d_rest, const double* d_scalar,
                       float* d_points, float* d_values, cudaStream_t stream) {
  if (n_vertices <= 0) {
    return;
  }
  gather_live_surface_kernel<<<Blocks(n_vertices), kThreads, 0, stream>>>(
      n_vertices, terms, d_coefs, d_weights, d_x12, d_y12, d_z12, d_rest,
      d_scalar, d_points, d_values);
}

void ColorizeLiveSurface(int n_vertices, const float* d_values, float color_max,
                         float* d_colors, cudaStream_t stream) {
  if (n_vertices <= 0) {
    return;
  }
  const float inv_max = color_max > 0.0f ? 1.0f / color_max : 0.0f;
  colorize_live_surface_kernel<<<Blocks(n_vertices), kThreads, 0, stream>>>(
      n_vertices, d_values, inv_max, d_colors);
}

}  // namespace LiveView
