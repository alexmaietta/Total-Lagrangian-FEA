#pragma once

#include <cuda_runtime.h>

// CUDA helpers for LiveViewer (see LiveViewer.h). Every surface vertex is a
// fixed weighted sum of solver coefficients (a node for FEAT10, shape
// functions times coefficients for ANCF), given by d_coefs and d_weights
// (n_vertices * terms each). Components are 0 for the magnitude of a vector,
// or 1, 2, 3 for x, y, z.

namespace LiveView {

/**
 * Evaluate surface vertices from the solver's coefficients.
 * @param d_x12 Coefficient x components
 * @param d_y12 Coefficient y components
 * @param d_z12 Coefficient z components
 * @param d_points Output vertices, [x0,y0,z0, ...]
 */
void GatherLivePoints(int n_vertices, int terms, const int* d_coefs,
                      const float* d_weights, const double* d_x12,
                      const double* d_y12, const double* d_z12, float* d_points,
                      cudaStream_t stream);

/**
 * Displacement of each vertex from its rest position.
 * @param d_points Vertices, [x0,y0,z0, ...]
 * @param d_rest Vertices at rest, [x0,y0,z0, ...]
 * @param d_values Output value of each vertex
 */
void LiveDisplacement(int n_vertices, const float* d_points,
                      const float* d_rest, int component, float* d_values,
                      cudaStream_t stream);

/**
 * A scalar per coefficient, read at each vertex's first coefficient.
 */
void LiveNodeScalar(int n_vertices, int terms, const int* d_coefs,
                    const double* d_scalar, float* d_values,
                    cudaStream_t stream);

/**
 * A vector per coefficient, interpolated to each vertex like its position.
 * Coefficient c's components are d_x[c * stride], d_y[c * stride] and
 * d_z[c * stride].
 */
void LiveNodeVector(int n_vertices, int terms, const int* d_coefs,
                    const float* d_weights, const double* d_x,
                    const double* d_y, const double* d_z, int stride,
                    int component, float* d_values, cudaStream_t stream);

/**
 * A scalar per element, averaged over the elements each vertex lies in
 * (d_elements[d_offsets[i]] up to d_elements[d_offsets[i + 1]] for vertex i).
 */
void LiveElementScalar(int n_vertices, const int* d_offsets,
                       const int* d_elements, const double* d_scalar,
                       float* d_values, cudaStream_t stream);

/**
 * Widen a running range to the finite values given.
 * @param d_range [min, max], start it at [FLT_MAX, -FLT_MAX]
 */
void WidenLiveRange(int n_vertices, const float* d_values, float* d_range,
                    cudaStream_t stream);

/**
 * Color values through the Turbo colormap, from lo at the bottom to hi at
 * the top.
 * @param d_colors Output colors, [r0,g0,b0, ...]
 */
void ColorizeLiveSurface(int n_vertices, const float* d_values, float lo,
                         float hi, float* d_colors, cudaStream_t stream);

}  // namespace LiveView
