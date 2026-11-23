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

// The magnitude of (x, y, z), or one of its components
__device__ float pick_component(double x, double y, double z, int component) {
  switch (component) {
    case 1:
      return static_cast<float>(x);
    case 2:
      return static_cast<float>(y);
    case 3:
      return static_cast<float>(z);
    default:
      return static_cast<float>(sqrt(x * x + y * y + z * z));
  }
}

// Atomic min and max of floats through their bit patterns, which order like
// signed integers for non-negative floats and reversed for negative ones
__device__ void atomic_min_float(float* address, float value) {
  if (value >= 0.0f) {
    atomicMin(reinterpret_cast<int*>(address), __float_as_int(value));
  } else {
    atomicMax(reinterpret_cast<unsigned int*>(address), __float_as_uint(value));
  }
}

__device__ void atomic_max_float(float* address, float value) {
  if (value >= 0.0f) {
    atomicMax(reinterpret_cast<int*>(address), __float_as_int(value));
  } else {
    atomicMin(reinterpret_cast<unsigned int*>(address), __float_as_uint(value));
  }
}

__global__ void gather_live_points_kernel(
    int n_vertices, int terms, const int* d_coefs, const float* d_weights,
    const double* d_x12, const double* d_y12, const double* d_z12,
    float* d_points) {
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
}

__global__ void live_displacement_kernel(int n_vertices, const float* d_points,
                                         const float* d_rest, int component,
                                         float* d_values) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_vertices) {
    return;
  }
  d_values[i] =
      pick_component(d_points[3 * i + 0] - d_rest[3 * i + 0],
                     d_points[3 * i + 1] - d_rest[3 * i + 1],
                     d_points[3 * i + 2] - d_rest[3 * i + 2], component);
}

__global__ void live_node_scalar_kernel(int n_vertices, int terms,
                                        const int* d_coefs,
                                        const double* d_scalar,
                                        float* d_values) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_vertices) {
    return;
  }
  d_values[i] = static_cast<float>(d_scalar[d_coefs[i * terms]]);
}

__global__ void live_node_vector_kernel(int n_vertices, int terms,
                                        const int* d_coefs,
                                        const float* d_weights,
                                        const double* d_x, const double* d_y,
                                        const double* d_z, int stride,
                                        int component, float* d_values) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_vertices) {
    return;
  }
  double vx = 0.0, vy = 0.0, vz = 0.0;
  for (int k = 0; k < terms; ++k) {
    const size_t c = size_t(d_coefs[i * terms + k]) * stride;
    const double w = d_weights[i * terms + k];
    vx += w * d_x[c];
    vy += w * d_y[c];
    vz += w * d_z[c];
  }
  d_values[i] = pick_component(vx, vy, vz, component);
}

__global__ void live_element_scalar_kernel(int n_vertices, const int* d_offsets,
                                           const int* d_elements,
                                           const double* d_scalar,
                                           float* d_values) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_vertices) {
    return;
  }
  const int begin = d_offsets[i], end = d_offsets[i + 1];
  double sum = 0.0;
  for (int k = begin; k < end; ++k) {
    sum += d_scalar[d_elements[k]];
  }
  d_values[i] = end > begin ? static_cast<float>(sum / (end - begin)) : 0.0f;
}

__global__ void widen_live_range_kernel(int n_vertices, const float* d_values,
                                        float* d_range) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float lo = 3.4e38f, hi = -3.4e38f;
  if (i < n_vertices && isfinite(d_values[i])) {
    lo = hi = d_values[i];
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    lo = fminf(lo, __shfl_down_sync(0xffffffffu, lo, offset));
    hi = fmaxf(hi, __shfl_down_sync(0xffffffffu, hi, offset));
  }
  if ((threadIdx.x & 31) == 0 && lo <= hi) {
    atomic_min_float(&d_range[0], lo);
    atomic_max_float(&d_range[1], hi);
  }
}

__global__ void colorize_live_surface_kernel(int n_vertices,
                                             const float* d_values, float lo,
                                             float inv_span, float* d_colors) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_vertices) {
    return;
  }
  const float3 c      = turbo_colormap((d_values[i] - lo) * inv_span);
  d_colors[3 * i + 0] = c.x;
  d_colors[3 * i + 1] = c.y;
  d_colors[3 * i + 2] = c.z;
}

// A multiple of the warp size, which the range reduction relies on
constexpr int kThreads = 256;

int Blocks(int n) {
  return (n + kThreads - 1) / kThreads;
}

}  // namespace

void GatherLivePoints(int n_vertices, int terms, const int* d_coefs,
                      const float* d_weights, const double* d_x12,
                      const double* d_y12, const double* d_z12, float* d_points,
                      cudaStream_t stream) {
  if (n_vertices <= 0) {
    return;
  }
  gather_live_points_kernel<<<Blocks(n_vertices), kThreads, 0, stream>>>(
      n_vertices, terms, d_coefs, d_weights, d_x12, d_y12, d_z12, d_points);
}

void LiveDisplacement(int n_vertices, const float* d_points,
                      const float* d_rest, int component, float* d_values,
                      cudaStream_t stream) {
  if (n_vertices <= 0) {
    return;
  }
  live_displacement_kernel<<<Blocks(n_vertices), kThreads, 0, stream>>>(
      n_vertices, d_points, d_rest, component, d_values);
}

void LiveNodeScalar(int n_vertices, int terms, const int* d_coefs,
                    const double* d_scalar, float* d_values,
                    cudaStream_t stream) {
  if (n_vertices <= 0) {
    return;
  }
  live_node_scalar_kernel<<<Blocks(n_vertices), kThreads, 0, stream>>>(
      n_vertices, terms, d_coefs, d_scalar, d_values);
}

void LiveNodeVector(int n_vertices, int terms, const int* d_coefs,
                    const float* d_weights, const double* d_x,
                    const double* d_y, const double* d_z, int stride,
                    int component, float* d_values, cudaStream_t stream) {
  if (n_vertices <= 0) {
    return;
  }
  live_node_vector_kernel<<<Blocks(n_vertices), kThreads, 0, stream>>>(
      n_vertices, terms, d_coefs, d_weights, d_x, d_y, d_z, stride, component,
      d_values);
}

void LiveElementScalar(int n_vertices, const int* d_offsets,
                       const int* d_elements, const double* d_scalar,
                       float* d_values, cudaStream_t stream) {
  if (n_vertices <= 0) {
    return;
  }
  live_element_scalar_kernel<<<Blocks(n_vertices), kThreads, 0, stream>>>(
      n_vertices, d_offsets, d_elements, d_scalar, d_values);
}

void WidenLiveRange(int n_vertices, const float* d_values, float* d_range,
                    cudaStream_t stream) {
  if (n_vertices <= 0) {
    return;
  }
  widen_live_range_kernel<<<Blocks(n_vertices), kThreads, 0, stream>>>(
      n_vertices, d_values, d_range);
}

void ColorizeLiveSurface(int n_vertices, const float* d_values, float lo,
                         float hi, float* d_colors, cudaStream_t stream) {
  if (n_vertices <= 0) {
    return;
  }
  const float inv_span = hi > lo ? 1.0f / (hi - lo) : 0.0f;
  colorize_live_surface_kernel<<<Blocks(n_vertices), kThreads, 0, stream>>>(
      n_vertices, d_values, lo, inv_span, d_colors);
}

}  // namespace LiveView
