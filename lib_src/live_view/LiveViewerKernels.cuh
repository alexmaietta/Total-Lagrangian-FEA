#pragma once

#include <cuda_runtime.h>

// CUDA helpers for LiveViewer (see LiveViewer.h).

namespace LiveView {

/**
 * Evaluate surface vertices as fixed weighted sums of solver coefficients
 * (a node position for FEAT10, shape functions times coefficients for ANCF),
 * with the value each vertex is colored by. The value is a node scalar if
 * given (at the vertex's first coefficient), or else the vertex's
 * displacement from its rest position.
 * @param n_vertices Number of surface vertices
 * @param terms Coefficients per vertex
 * @param d_coefs Coefficient indices (n_vertices * terms)
 * @param d_weights Coefficient weights (n_vertices * terms)
 * @param d_x12 Coefficient x components
 * @param d_y12 Coefficient y components
 * @param d_z12 Coefficient z components
 * @param d_rest Rest position of each vertex, [x0,y0,z0, x1,y1,z1, ...]
 * @param d_scalar Node scalar to color by, or nullptr
 * @param d_points Output vertices, [x0,y0,z0, ...]
 * @param d_values Output value of each vertex (length n_vertices)
 * @param stream Stream the kernel is queued on
 */
void GatherLiveSurface(int n_vertices, int terms, const int* d_coefs,
                       const float* d_weights, const double* d_x12,
                       const double* d_y12, const double* d_z12,
                       const float* d_rest, const double* d_scalar,
                       float* d_points, float* d_values, cudaStream_t stream);

/**
 * Color values through the Turbo colormap.
 * @param n_vertices Number of values
 * @param d_values Values (length n_vertices)
 * @param color_max Value mapped to the top of the colormap
 * @param d_colors Output colors, [r0,g0,b0, ...]
 * @param stream Stream the kernel is queued on
 */
void ColorizeLiveSurface(int n_vertices, const float* d_values, float color_max,
                         float* d_colors, cudaStream_t stream);

}  // namespace LiveView
