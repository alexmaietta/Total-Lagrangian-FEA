#include "ANCFSurfaces.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <utility>
#include <vector>

#include "../../lib_utils/cpu_utils.h"

namespace LiveView {
namespace {

// Monomial bases of the elements (as ancf3243_b_vec and ancf3443_b_vec)
Eigen::VectorXd Basis3243(double u, double v, double w) {
  Eigen::VectorXd b(8);
  b << 1.0, u, v, w, u * v, u * w, u * u, u * u * u;
  return b;
}

Eigen::VectorXd Basis3443(double u, double v, double w) {
  Eigen::VectorXd b(16);
  b << 1.0, u, v, w, u * v, u * w, v * w, u * v * w, u * u, v * v, u * u * v,
      u * v * v, u * u * u, v * v * v, u * u * u * v, u * v * v * v;
  return b;
}

/**
 * Adds vertices of one element at element coordinates, as weighted sums of
 * its coefficients (s = B_inv b, as in the element kernels).
 */
class ElementSampler {
 public:
  ElementSampler(LinearSurface& out, const Eigen::VectorXd& x12,
                 const Eigen::VectorXd& y12, const Eigen::VectorXd& z12,
                 const Eigen::MatrixXd& B_inv,
                 Eigen::VectorXd (*basis)(double, double, double))
      : out_(out),
        x12_(x12),
        y12_(y12),
        z12_(z12),
        B_inv_(B_inv),
        basis_(basis) {}

  // Coefficient i of the element is dof i % 4 of its node i / 4
  void SetElement(const Eigen::MatrixXi& connectivity, int element) {
    coefs_.resize(B_inv_.rows());
    for (int i = 0; i < int(coefs_.size()); ++i) {
      coefs_[i] = connectivity(element, i / 4) * 4 + i % 4;
    }
  }

  Eigen::Vector3d Position(double u, double v, double w) const {
    const Eigen::VectorXd s = B_inv_ * basis_(u, v, w);
    Eigen::Vector3d p       = Eigen::Vector3d::Zero();
    for (int i = 0; i < int(coefs_.size()); ++i) {
      p += s(i) *
           Eigen::Vector3d(x12_(coefs_[i]), y12_(coefs_[i]), z12_(coefs_[i]));
    }
    return p;
  }

  int Add(double u, double v, double w) {
    const Eigen::VectorXd s = B_inv_ * basis_(u, v, w);
    for (int i = 0; i < int(coefs_.size()); ++i) {
      out_.coefs.push_back(coefs_[i]);
      out_.weights.push_back(float(s(i)));
    }
    out_.rest.push_back(Position(u, v, w));
    return int(out_.rest.size()) - 1;
  }

  // Two triangles over a quad given in order around it
  void Quad(int a, int b, int c, int d) {
    out_.triangles.emplace_back(a, b, c);
    out_.triangles.emplace_back(a, c, d);
  }

  // A grid of (n_a + 1) x (n_b + 1) vertices over the bilinear patch through
  // the four corner coordinates, given in order around it (n_a along the
  // first edge)
  void Patch(const std::array<Eigen::Vector3d, 4>& corner, int n_a, int n_b) {
    std::vector<int> ids;
    for (int j = 0; j <= n_b; ++j) {
      for (int i = 0; i <= n_a; ++i) {
        const double a = double(i) / n_a, b = double(j) / n_b;
        const Eigen::Vector3d c = (1 - a) * (1 - b) * corner[0] +
                                  a * (1 - b) * corner[1] + a * b * corner[2] +
                                  (1 - a) * b * corner[3];
        ids.push_back(Add(c.x(), c.y(), c.z()));
      }
    }
    for (int j = 0; j < n_b; ++j) {
      for (int i = 0; i < n_a; ++i) {
        const int k = j * (n_a + 1) + i;
        Quad(ids[k], ids[k + 1], ids[k + n_a + 2], ids[k + n_a + 1]);
      }
    }
  }

  // Node position (coefficient 0 of the node)
  Eigen::Vector3d Node(int node) const {
    return {x12_(node * 4), y12_(node * 4), z12_(node * 4)};
  }

 private:
  LinearSurface& out_;
  const Eigen::VectorXd& x12_;
  const Eigen::VectorXd& y12_;
  const Eigen::VectorXd& z12_;
  const Eigen::MatrixXd& B_inv_;
  Eigen::VectorXd (*basis_)(double, double, double);
  std::vector<int> coefs_;
};

// Element node nearest a point, among the element's nodes
int NearestNode(const ElementSampler& sampler,
                const Eigen::MatrixXi& connectivity, int element,
                const Eigen::Vector3d& p, double* distance) {
  int best       = connectivity(element, 0);
  double nearest = (sampler.Node(best) - p).norm();
  for (int j = 1; j < connectivity.cols(); ++j) {
    const int node = connectivity(element, j);
    const double d = (sampler.Node(node) - p).norm();
    if (d < nearest) {
      nearest = d;
      best    = node;
    }
  }
  *distance = nearest;
  return best;
}

}  // namespace

LinearSurface ANCF3243Surface(const Eigen::VectorXd& x12,
                              const Eigen::VectorXd& y12,
                              const Eigen::VectorXd& z12,
                              const Eigen::MatrixXi& connectivity, double L,
                              double W, double H, int segments) {
  LinearSurface out;
  out.terms = 8;
  Eigen::MatrixXd B_inv;
  ANCFCPUUtils::ANCF3243_B12_matrix(L, W, H, B_inv, 8);
  ElementSampler sampler(out, x12, y12, z12, B_inv, Basis3243);

  std::map<int, int> uses;  // Elements at each node
  for (int e = 0; e < connectivity.rows(); ++e) {
    for (int j = 0; j < 2; ++j) {
      ++uses[connectivity(e, j)];
    }
  }

  // Cross-section corners, in order around it
  const double v = 0.5 * W, w = 0.5 * H;
  const std::array<std::pair<double, double>, 4> section = {
      {{-v, -w}, {v, -w}, {v, w}, {-v, w}}};
  double worst = 0.0;
  for (int e = 0; e < connectivity.rows(); ++e) {
    sampler.SetElement(connectivity, e);
    for (int f = 0; f < 4; ++f) {
      const auto [v0, w0] = section[f];
      const auto [v1, w1] = section[(f + 1) % 4];
      sampler.Patch(
          {Eigen::Vector3d(-0.5 * L, v0, w0), Eigen::Vector3d(0.5 * L, v0, w0),
           Eigen::Vector3d(0.5 * L, v1, w1), Eigen::Vector3d(-0.5 * L, v1, w1)},
          segments, 1);
    }
    // Caps at ends no other element shares
    for (const double u : {-0.5 * L, 0.5 * L}) {
      double distance = 0.0;
      const int node  = NearestNode(sampler, connectivity, e,
                                    sampler.Position(u, 0, 0), &distance);
      worst           = std::max(worst, distance);
      if (uses[node] == 1) {
        std::array<int, 4> ids;
        for (int c = 0; c < 4; ++c) {
          ids[c] = sampler.Add(u, section[c].first, section[c].second);
        }
        sampler.Quad(ids[0], ids[1], ids[2], ids[3]);
      }
    }
  }
  if (worst > 1e-6 * std::max(L, 1e-12)) {
    std::printf(
        "ANCF3243Surface: element ends are %g from their nodes; the shape "
        "functions may not match the element's\n",
        worst);
  }
  return out;
}

LinearSurface ANCF3443Surface(const Eigen::VectorXd& x12,
                              const Eigen::VectorXd& y12,
                              const Eigen::VectorXd& z12,
                              const Eigen::MatrixXi& connectivity, double L,
                              double W, double H, int segments) {
  LinearSurface out;
  out.terms = 16;
  Eigen::MatrixXd B_inv;
  ANCFCPUUtils::ANCF3443_B12_matrix(L, W, H, B_inv, 16);
  ElementSampler sampler(out, x12, y12, z12, B_inv, Basis3443);

  const double u = 0.5 * L, v = 0.5 * W, w = 0.5 * H;
  // Element corners in order around it, and the nodes there
  const std::array<std::pair<double, double>, 4> corners = {
      {{-u, -v}, {u, -v}, {u, v}, {-u, v}}};
  std::vector<std::array<int, 4>> corner_nodes(connectivity.rows());
  std::map<std::pair<int, int>, int> edge_uses;
  double worst = 0.0;
  for (int e = 0; e < connectivity.rows(); ++e) {
    sampler.SetElement(connectivity, e);
    for (int c = 0; c < 4; ++c) {
      double distance    = 0.0;
      corner_nodes[e][c] = NearestNode(
          sampler, connectivity, e,
          sampler.Position(corners[c].first, corners[c].second, 0), &distance);
      worst = std::max(worst, distance);
    }
    for (int c = 0; c < 4; ++c) {
      const int a = corner_nodes[e][c], b = corner_nodes[e][(c + 1) % 4];
      ++edge_uses[{std::min(a, b), std::max(a, b)}];
    }
  }

  for (int e = 0; e < connectivity.rows(); ++e) {
    sampler.SetElement(connectivity, e);
    for (const double side : {-w, w}) {
      sampler.Patch(
          {Eigen::Vector3d(-u, -v, side), Eigen::Vector3d(u, -v, side),
           Eigen::Vector3d(u, v, side), Eigen::Vector3d(-u, v, side)},
          segments, segments);
    }
    // Side strips along edges no other element shares
    for (int c = 0; c < 4; ++c) {
      const int a = corner_nodes[e][c], b = corner_nodes[e][(c + 1) % 4];
      if (edge_uses[{std::min(a, b), std::max(a, b)}] != 1) {
        continue;
      }
      const auto [u0, v0] = corners[c];
      const auto [u1, v1] = corners[(c + 1) % 4];
      sampler.Patch({Eigen::Vector3d(u0, v0, -w), Eigen::Vector3d(u1, v1, -w),
                     Eigen::Vector3d(u1, v1, w), Eigen::Vector3d(u0, v0, w)},
                    segments, 1);
    }
  }
  if (worst > 1e-6 * std::max(L, 1e-12)) {
    std::printf(
        "ANCF3443Surface: element corners are %g from their nodes; the shape "
        "functions may not match the element's\n",
        worst);
  }
  return out;
}

}  // namespace LiveView
