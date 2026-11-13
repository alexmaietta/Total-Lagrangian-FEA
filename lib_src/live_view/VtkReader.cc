#include "VtkReader.h"

#include <glob.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace LiveView {
namespace {

// Value of attribute 'name' within the tag text, or empty.
std::string Attribute(const std::string& tag, const std::string& name) {
  const std::string key = " " + name + "=\"";
  const size_t at       = tag.find(key);
  if (at == std::string::npos) {
    return {};
  }
  const size_t begin = at + key.size();
  const size_t end   = tag.find('"', begin);
  return end == std::string::npos ? std::string()
                                  : tag.substr(begin, end - begin);
}

// Parse whitespace-separated numbers in [begin, end).
void ParseNumbers(const char* begin, const char* end,
                  std::vector<double>& values) {
  const char* p = begin;
  while (p < end) {
    while (p < end && std::isspace(static_cast<unsigned char>(*p))) {
      ++p;
    }
    if (p >= end) {
      break;
    }
    char* next     = nullptr;
    const double v = std::strtod(p, &next);
    if (next == p) {
      throw std::runtime_error("unexpected text in DataArray");
    }
    values.push_back(v);
    p = next;
  }
}

template <class T>
std::vector<T> Cast(const std::vector<double>& values) {
  std::vector<T> out(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    out[i] = static_cast<T>(values[i]);
  }
  return out;
}

// Last number in a file name, for sorting frames.
long FrameNumber(const std::string& path) {
  const std::string name = path.substr(path.find_last_of('/') + 1);
  long number            = -1;
  for (size_t i = 0; i < name.size();) {
    if (std::isdigit(static_cast<unsigned char>(name[i]))) {
      number = std::strtol(name.c_str() + i, nullptr, 10);
      while (i < name.size() &&
             std::isdigit(static_cast<unsigned char>(name[i]))) {
        ++i;
      }
    } else {
      ++i;
    }
  }
  return number;
}

}  // namespace

const VtkArray* VtkPiece::PointArray(const std::string& name) const {
  for (const VtkArray& a : point_data) {
    if (a.name == name) {
      return &a;
    }
  }
  return nullptr;
}

const VtkArray* VtkPiece::CellArray(const std::string& name) const {
  for (const VtkArray& a : cell_data) {
    if (a.name == name) {
      return &a;
    }
  }
  return nullptr;
}

namespace {

// VTK XML (.vtu, .vtp). Each DataArray belongs to the section opened most
// recently before it.
void ReadXml(const std::string& text, VtkPiece& piece) {
  {
    enum class Section { None, Points, Cells, Polys, PointData, CellData };
    Section section = Section::None;
    size_t at       = 0;
    while ((at = text.find('<', at)) != std::string::npos) {
      const size_t close = text.find('>', at);
      if (close == std::string::npos) {
        break;
      }
      const std::string tag = text.substr(at, close - at + 1);
      at                    = close + 1;

      if (tag.rfind("<Piece", 0) == 0) {
        piece.n_points = std::atoi(Attribute(tag, "NumberOfPoints").c_str());
      } else if (tag.rfind("<Points", 0) == 0) {
        section = Section::Points;
      } else if (tag.rfind("<Cells", 0) == 0) {
        section = Section::Cells;
      } else if (tag.rfind("<Polys", 0) == 0) {
        section = Section::Polys;
      } else if (tag.rfind("<PointData", 0) == 0) {
        section = Section::PointData;
      } else if (tag.rfind("<CellData", 0) == 0) {
        section = Section::CellData;
      } else if (tag.rfind("</", 0) == 0 && tag.rfind("</DataArray", 0) != 0) {
        section = Section::None;
      } else if (tag.rfind("<DataArray", 0) == 0) {
        const std::string format = Attribute(tag, "format");
        if (!format.empty() && format != "ascii") {
          throw std::runtime_error("DataArray format \"" + format +
                                   "\" is not supported (ascii only)");
        }
        const size_t end = text.find("</DataArray>", at);
        if (end == std::string::npos) {
          throw std::runtime_error("unterminated DataArray");
        }
        VtkArray array;
        array.name                   = Attribute(tag, "Name");
        const std::string components = Attribute(tag, "NumberOfComponents");
        array.components =
            components.empty() ? 1 : std::atoi(components.c_str());
        ParseNumbers(text.data() + at, text.data() + end, array.values);
        at = end + std::strlen("</DataArray>");

        switch (section) {
          case Section::Points:
            piece.points = std::move(array.values);
            break;
          case Section::Cells:
          case Section::Polys:
            if (array.name == "connectivity") {
              piece.connectivity = Cast<int32_t>(array.values);
            } else if (array.name == "offsets") {
              piece.offsets = Cast<int32_t>(array.values);
            } else if (array.name == "types") {
              piece.types = Cast<int32_t>(array.values);
            }
            break;
          case Section::PointData:
            piece.point_data.push_back(std::move(array));
            break;
          case Section::CellData:
            piece.cell_data.push_back(std::move(array));
            break;
          case Section::None:
            break;
        }
      }
    }
  }
}

// Whitespace-separated tokens of a legacy VTK file
class Tokens {
 public:
  explicit Tokens(const std::string& text)
      : p_(text.data()), end_(text.data() + text.size()) {}

  bool Done() {
    Skip();
    return p_ >= end_;
  }

  std::string Word() {
    Skip();
    const char* begin = p_;
    while (p_ < end_ && !std::isspace(static_cast<unsigned char>(*p_))) {
      ++p_;
    }
    if (begin == p_) {
      throw std::runtime_error("unexpected end of file");
    }
    return std::string(begin, p_);
  }

  // The next word, without consuming it
  std::string Peek() {
    const char* saved = p_;
    std::string word  = Done() ? std::string() : Word();
    p_                = saved;
    return word;
  }

  long Integer() {
    return std::strtol(Word().c_str(), nullptr, 10);
  }

  void Numbers(size_t count, std::vector<double>& values) {
    values.reserve(values.size() + count);
    for (size_t i = 0; i < count; ++i) {
      Skip();
      char* next     = nullptr;
      const double v = std::strtod(p_, &next);
      if (next == p_) {
        throw std::runtime_error("expected a number");
      }
      values.push_back(v);
      p_ = next;
    }
  }

  // Rest of the current line (for the title)
  void SkipLine() {
    while (p_ < end_ && *p_ != '\n') {
      ++p_;
    }
    if (p_ < end_) {
      ++p_;
    }
  }

 private:
  void Skip() {
    while (p_ < end_ && std::isspace(static_cast<unsigned char>(*p_))) {
      ++p_;
    }
  }

  const char* p_;
  const char* end_;
};

// Cells or polygons as [count, ids...] rows, into connectivity and offsets
void ReadCellRows(Tokens& tokens, VtkPiece& piece, bool keep) {
  const long n_cells = tokens.Integer();
  const long size    = tokens.Integer();
  std::vector<double> values;
  tokens.Numbers(size_t(size), values);
  if (!keep) {
    return;
  }
  size_t at = 0;
  for (long c = 0; c < n_cells; ++c) {
    const int count = int(values.at(at++));
    for (int j = 0; j < count; ++j) {
      piece.connectivity.push_back(int32_t(values.at(at++)));
    }
    piece.offsets.push_back(int32_t(piece.connectivity.size()));
  }
}

// Legacy VTK (.vtk), ASCII only
void ReadLegacy(const std::string& text, VtkPiece& piece) {
  Tokens tokens(text);
  tokens.SkipLine();  // # vtk DataFile Version x.y
  tokens.SkipLine();  // Title
  const std::string encoding = tokens.Word();
  if (encoding != "ASCII") {
    throw std::runtime_error("legacy VTK encoding " + encoding +
                             " is not supported (ASCII only)");
  }
  if (tokens.Word() != "DATASET") {
    throw std::runtime_error("expected DATASET");
  }
  const std::string dataset = tokens.Word();
  if (dataset != "UNSTRUCTURED_GRID" && dataset != "POLYDATA") {
    throw std::runtime_error("legacy VTK dataset " + dataset +
                             " is not supported");
  }

  std::vector<VtkArray>* data = nullptr;  // POINT_DATA or CELL_DATA
  long data_count             = 0;
  while (!tokens.Done()) {
    const std::string key = tokens.Word();
    if (key == "POINTS") {
      piece.n_points = int(tokens.Integer());
      tokens.Word();  // Type
      tokens.Numbers(size_t(piece.n_points) * 3, piece.points);
    } else if (key == "CELLS" || key == "POLYGONS") {
      ReadCellRows(tokens, piece, true);
    } else if (key == "VERTICES" || key == "LINES" ||
               key == "TRIANGLE_STRIPS") {
      ReadCellRows(tokens, piece, false);
    } else if (key == "CELL_TYPES") {
      std::vector<double> types;
      tokens.Numbers(size_t(tokens.Integer()), types);
      piece.types = Cast<int32_t>(types);
    } else if (key == "POINT_DATA" || key == "CELL_DATA") {
      data       = key == "POINT_DATA" ? &piece.point_data : &piece.cell_data;
      data_count = tokens.Integer();
    } else if (key == "SCALARS" || key == "VECTORS" || key == "NORMALS" ||
               key == "TENSORS") {
      if (!data) {
        throw std::runtime_error(key + " before POINT_DATA or CELL_DATA");
      }
      VtkArray array;
      array.name = tokens.Word();
      tokens.Word();  // Type
      array.components = key == "SCALARS" ? 1 : (key == "TENSORS" ? 9 : 3);
      if (key == "SCALARS") {
        const std::string next = tokens.Peek();
        if (!next.empty() &&
            std::isdigit(static_cast<unsigned char>(next[0]))) {
          array.components = int(tokens.Integer());
        }
        if (tokens.Peek() == "LOOKUP_TABLE") {
          tokens.Word();
          tokens.Word();  // Table name
        }
      }
      tokens.Numbers(size_t(data_count) * array.components, array.values);
      data->push_back(std::move(array));
    } else if (key == "LOOKUP_TABLE") {
      tokens.Word();  // Table name
      std::vector<double> table;
      tokens.Numbers(size_t(tokens.Integer()) * 4, table);
    } else if (key == "FIELD") {
      if (!data) {
        throw std::runtime_error("FIELD before POINT_DATA or CELL_DATA");
      }
      tokens.Word();  // Field name
      const long n_arrays = tokens.Integer();
      for (long a = 0; a < n_arrays; ++a) {
        VtkArray array;
        array.name        = tokens.Word();
        array.components  = int(tokens.Integer());
        const long tuples = tokens.Integer();
        tokens.Word();  // Type
        tokens.Numbers(size_t(tuples) * array.components, array.values);
        data->push_back(std::move(array));
      }
    } else {
      throw std::runtime_error("unsupported legacy VTK section " + key);
    }
  }
}

}  // namespace

bool ReadVtkPiece(const std::string& path, VtkPiece& piece,
                  std::string* error) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    *error = "cannot open " + path;
    return false;
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  const std::string text = buffer.str();

  piece = {};
  try {
    if (text.rfind("# vtk DataFile", 0) == 0) {
      ReadLegacy(text, piece);
    } else {
      ReadXml(text, piece);
    }
  } catch (const std::exception& e) {
    *error = path + ": " + e.what();
    return false;
  }
  if (piece.points.size() != size_t(piece.n_points) * 3) {
    *error = path + ": expected " + std::to_string(piece.n_points) +
             " points, read " + std::to_string(piece.points.size() / 3);
    return false;
  }
  return true;
}

std::vector<std::string> ListFrames(const std::string& pattern) {
  std::vector<std::string> files;
  glob_t found;
  if (glob(pattern.c_str(), 0, nullptr, &found) == 0) {
    for (size_t i = 0; i < found.gl_pathc; ++i) {
      files.emplace_back(found.gl_pathv[i]);
    }
  }
  globfree(&found);
  std::stable_sort(files.begin(), files.end(),
                   [](const std::string& a, const std::string& b) {
                     return FrameNumber(a) < FrameNumber(b);
                   });
  return files;
}

VtkSequence::VtkSequence(std::vector<std::string> files, std::string scalar)
    : files_(std::move(files)), scalar_(std::move(scalar)) {
  if (files_.empty()) {
    throw std::runtime_error("no frames to play");
  }
  VtkPiece first;
  std::string error;
  if (!ReadVtkPiece(files_[0], first, &error)) {
    throw std::runtime_error(error);
  }
  n_nodes_ = first.n_points;

  // Quadratic tets only (VTK type 24, the FEAT10 node order)
  const size_t n_cells = first.offsets.size();
  for (size_t c = 0; c < n_cells; ++c) {
    const int32_t begin = c == 0 ? 0 : first.offsets[c - 1];
    if ((!first.types.empty() && first.types[c] != 24) ||
        first.offsets[c] - begin != 10) {
      throw std::runtime_error(files_[0] +
                               " has cells other than quadratic tets (VTK "
                               "type 24), which playback doesn't support yet");
    }
  }
  elements_.resize(Eigen::Index(n_cells), 10);
  elements_per_node_.assign(n_nodes_, 0);
  for (size_t c = 0; c < n_cells; ++c) {
    for (int j = 0; j < 10; ++j) {
      const int node                = first.connectivity[c * 10 + j];
      elements_(Eigen::Index(c), j) = node;
      elements_per_node_.at(node) += 1;
    }
  }

  // Points are deformed, so the rest positions are points minus displacement
  const VtkArray* displacement = first.PointArray("displacement");
  rest_nodes_.resize(n_nodes_, 3);
  for (int i = 0; i < n_nodes_; ++i) {
    for (int a = 0; a < 3; ++a) {
      const double d    = displacement && displacement->components == 3
                              ? displacement->values[size_t(i) * 3 + a]
                              : 0.0;
      rest_nodes_(i, a) = first.points[size_t(i) * 3 + a] - d;
    }
  }

  for (const VtkArray& a : first.point_data) {
    fields_.push_back(a.name);
  }
  for (const VtkArray& a : first.cell_data) {
    fields_.push_back(a.name);
  }
  if (!scalar_.empty() && !HasField(scalar_)) {
    throw std::runtime_error(files_[0] + " has no field \"" + scalar_ + "\"");
  }
  cache_.resize(files_.size());
}

bool VtkSequence::HasField(const std::string& name) const {
  return std::find(fields_.begin(), fields_.end(), name) != fields_.end();
}

void VtkSequence::Parse(int frame, Cached& cached) const {
  VtkPiece piece;
  std::string error;
  if (!ReadVtkPiece(files_.at(frame), piece, &error)) {
    throw std::runtime_error(error);
  }
  if (piece.n_points != n_nodes_) {
    throw std::runtime_error(files_[frame] + " has " +
                             std::to_string(piece.n_points) + " points, " +
                             files_[0] + " has " + std::to_string(n_nodes_));
  }
  cached.xyz.resize(size_t(n_nodes_) * 3);
  for (int i = 0; i < n_nodes_; ++i) {
    for (int a = 0; a < 3; ++a) {
      cached.xyz[size_t(a) * n_nodes_ + i] =
          float(piece.points[size_t(i) * 3 + a]);
    }
  }

  cached.scalar.clear();
  if (scalar_.empty()) {
    return;
  }
  cached.scalar.assign(n_nodes_, 0.0f);
  if (const VtkArray* point = piece.PointArray(scalar_)) {
    for (int i = 0; i < n_nodes_; ++i) {
      cached.scalar[i] = float(point->values[size_t(i) * point->components]);
    }
  } else if (const VtkArray* cell = piece.CellArray(scalar_)) {
    // Average the element values onto their nodes
    for (Eigen::Index c = 0; c < elements_.rows(); ++c) {
      const float v = float(cell->values[size_t(c) * cell->components]);
      for (int j = 0; j < 10; ++j) {
        cached.scalar[elements_(c, j)] += v;
      }
    }
    for (int i = 0; i < n_nodes_; ++i) {
      if (elements_per_node_[i] > 0) {
        cached.scalar[i] /= float(elements_per_node_[i]);
      }
    }
  } else {
    throw std::runtime_error(files_[frame] + " has no field \"" + scalar_ +
                             "\"");
  }
}

void VtkSequence::Frame(int frame, std::vector<double>& xyz,
                        std::vector<double>& scalar) {
  std::optional<Cached>& cached = cache_.at(frame);
  if (!cached) {
    cached.emplace();
    Parse(frame, *cached);
  }
  xyz.assign(cached->xyz.begin(), cached->xyz.end());
  scalar.assign(cached->scalar.begin(), cached->scalar.end());
}

double VtkSequence::EstimateScalarMax(int samples) {
  if (scalar_.empty()) {
    return 0.0;
  }
  double max = 0.0;
  std::vector<double> xyz, scalar;
  const int n = std::min(samples, Size());
  for (int s = 0; s < n; ++s) {
    const int frame = n == 1 ? 0 : int((long(Size() - 1) * s) / (n - 1));
    Frame(frame, xyz, scalar);
    for (double v : scalar) {
      max = std::max(max, v);
    }
  }
  return max;
}

}  // namespace LiveView
