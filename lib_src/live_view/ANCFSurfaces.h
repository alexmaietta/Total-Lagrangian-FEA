#pragma once

#include <Eigen/Dense>

#include "LiveViewer.h"

// Surfaces of ANCF elements for LiveViewer. ANCF positions are linear in the
// nodal coefficients, r(u, v, w) = sum_i s_i(u, v, w) e_i, so a point at fixed
// element coordinates is a fixed weighted sum of coefficients. These builders
// sample the faces of each element and keep the shape function values as
// weights, so the surfaces follow the elements exactly, bending included.

namespace LiveView {

/**
 * Surface of ANCF3243 beam elements: the four sides of each element's box
 * cross-section along its length, and end caps where an element ends at a
 * node no other element shares.
 * @param x12 Rest coefficient x components (4 per node)
 * @param y12 Rest coefficient y components
 * @param z12 Rest coefficient z components
 * @param connectivity Node IDs of each element (n_elements x 2)
 * @param L Element length
 * @param W Cross-section width
 * @param H Cross-section height
 * @param segments Samples along each element's length
 */
LinearSurface ANCF3243Surface(const Eigen::VectorXd& x12,
                              const Eigen::VectorXd& y12,
                              const Eigen::VectorXd& z12,
                              const Eigen::MatrixXi& connectivity, double L,
                              double W, double H, int segments = 4);

/**
 * Surface of ANCF3443 shell elements: the top and bottom faces of each
 * element, and side strips along edges no other element shares.
 * @param x12 Rest coefficient x components (4 per node)
 * @param y12 Rest coefficient y components
 * @param z12 Rest coefficient z components
 * @param connectivity Node IDs of each element (n_elements x 4)
 * @param L Element length
 * @param W Element width
 * @param H Shell thickness
 * @param segments Samples along each element edge
 */
LinearSurface ANCF3443Surface(const Eigen::VectorXd& x12,
                              const Eigen::VectorXd& y12,
                              const Eigen::VectorXd& z12,
                              const Eigen::MatrixXi& connectivity, double L,
                              double W, double H, int segments = 4);

}  // namespace LiveView
