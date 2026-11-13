#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// Reader for the VTK files the demos and validation scripts write. VTK XML
// (.vtu, .vtp, see WriteOutputVTU and lib_utils/visualization_utils.h) with
// inline ASCII data arrays in a single Piece, and legacy ASCII VTK (.vtk, see
// WriteOutputVTK and test-scripts/validation/fenics/tet10_mesh_utils.py).

namespace LiveView {

/**
 * One data array (point or cell field) of a VTK file
 */
struct VtkArray {
  std::string name;
  int components = 1;
  std::vector<double> values;  // (n_tuples * components)
};

/**
 * The single Piece of a .vtu or .vtp file, or the dataset of a .vtk file
 */
struct VtkPiece {
  int n_points = 0;
  std::vector<double> points;  // [x0,y0,z0, x1,y1,z1, ...]
  // Cells (.vtu) or polygons (.vtp)
  std::vector<int32_t> connectivity;
  std::vector<int32_t> offsets;
  std::vector<int32_t> types;  // VTK cell types (.vtu only)
  std::vector<VtkArray> point_data;
  std::vector<VtkArray> cell_data;

  /**
   * Point or cell data array by name, or nullptr if there is none.
   */
  const VtkArray* PointArray(const std::string& name) const;
  const VtkArray* CellArray(const std::string& name) const;
};

/**
 * Read a .vtu, .vtp or legacy .vtk file (told apart by their contents).
 * @param path File to read
 * @param piece Filled with the file's contents
 * @param error Set to the reason when reading fails
 * @return true on success
 */
bool ReadVtkPiece(const std::string& path, VtkPiece& piece, std::string* error);

/**
 * Files matching a glob pattern, sorted by the last number in their names
 * (so frame 10 comes after frame 9).
 */
std::vector<std::string> ListFrames(const std::string& pattern);

/**
 * A sequence of FEAT10 frames (.vtu or .vtk) with constant connectivity,
 * played back as node positions and a node scalar. Frames without a
 * displacement field (e.g. the FEniCS reference runs) take the first frame as
 * the rest positions.
 */
class VtkSequence {
 public:
  /**
   * Read the first frame, for the mesh and the rest positions.
   * @param files Frame files in playback order
   * @param scalar Field to color by. "von_mises_stress" (per element, averaged
   * onto nodes), any per-node field, or empty for displacement
   */
  VtkSequence(std::vector<std::string> files, std::string scalar);

  int Size() const {
    return int(files_.size());
  }
  const std::string& File(int frame) const {
    return files_[frame];
  }
  int NumNodes() const {
    return n_nodes_;
  }
  int NumElements() const {
    return int(elements_.rows());
  }

  /**
   * Rest positions (frame 0 points minus its displacement), n_nodes x 3.
   */
  const Eigen::MatrixXd& RestNodes() const {
    return rest_nodes_;
  }

  /**
   * T10 connectivity, n_elements x 10.
   */
  const Eigen::MatrixXi& Elements() const {
    return elements_;
  }

  /**
   * Whether the first frame has a point or cell field of this name.
   */
  bool HasField(const std::string& name) const;

  /**
   * Node positions and scalar of a frame, parsed once and then cached.
   * @param frame Frame index
   * @param xyz Filled with [x...][y...][z...] (n_nodes each)
   * @param scalar Filled with one value per node (empty without a scalar)
   */
  void Frame(int frame, std::vector<double>& xyz, std::vector<double>& scalar);

  /**
   * Largest node scalar over a few frames spread through the sequence, for
   * the color range.
   */
  double EstimateScalarMax(int samples = 8);

 private:
  struct Cached {
    std::vector<float> xyz;
    std::vector<float> scalar;
  };

  void Parse(int frame, Cached& cached) const;

  std::vector<std::string> files_;
  std::string scalar_;
  int n_nodes_ = 0;
  Eigen::MatrixXd rest_nodes_;
  Eigen::MatrixXi elements_;
  std::vector<int> elements_per_node_;
  std::vector<std::string> fields_;
  std::vector<std::optional<Cached>> cache_;
};

}  // namespace LiveView
