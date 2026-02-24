#include <stdexcept>
#include <string>

#include <torch/extension.h>

#include "indexflat_c_api.h"

namespace py = pybind11;

namespace {

inline void check_rc(int rc) {
  if (rc == 0)
    return;
  const char *msg = indexflat_last_error();
  throw std::runtime_error(msg ? std::string(msg) : std::string("indexflat C API call failed"));
}

inline torch::Tensor require_cpu_f32_2d(const torch::Tensor &x, int dim) {
  if (!x.defined())
    throw std::invalid_argument("tensor is undefined");
  if (x.dim() != 2)
    throw std::invalid_argument("expected 2D tensor [N, D]");
  if (x.device().type() != torch::kCPU)
    throw std::invalid_argument("only CPU tensor is supported currently");
  if (x.size(1) != dim)
    throw std::invalid_argument("dimension mismatch");
  torch::Tensor y = x;
  if (y.scalar_type() != torch::kFloat32)
    y = y.to(torch::kFloat32);
  if (!y.is_contiguous())
    y = y.contiguous();
  return y;
}

class TorchIndexFlatL2 {
public:
  TorchIndexFlatL2(int dim, int dtype) : dim_(dim), dtype_(dtype) {
    handle_ = indexflat_create(dim_, dtype_);
    if (!handle_)
      check_rc(-1);
  }

  ~TorchIndexFlatL2() {
    if (handle_) {
      indexflat_destroy(handle_);
      handle_ = nullptr;
    }
  }

  void add(const torch::Tensor &x) {
    const auto y = require_cpu_f32_2d(x, dim_);
    check_rc(indexflat_add_f32(handle_, y.data_ptr<float>(), y.size(0)));
  }

  py::tuple search(const torch::Tensor &q, int k, int backend = INDEXFLAT_BACKEND_AUTO,
                   int block_n = 256, int block_d = 64, int num_threads = 0) const {
    const auto qy = require_cpu_f32_2d(q, dim_);
    if (k <= 0)
      throw std::invalid_argument("k must be > 0");
    auto out_d = torch::empty({qy.size(0), k}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    auto out_i = torch::empty({qy.size(0), k}, torch::dtype(torch::kInt32).device(torch::kCPU));
    check_rc(indexflat_search_f32(handle_, qy.data_ptr<float>(), qy.size(0), k,
                                  out_d.data_ptr<float>(),
                                  reinterpret_cast<int32_t *>(out_i.data_ptr<int>()), backend,
                                  block_n, block_d, num_threads));
    return py::make_tuple(out_d, out_i);
  }

  int dim() const {
    return indexflat_dim(handle_);
  }
  int64_t ntotal() const {
    return indexflat_ntotal(handle_);
  }

private:
  int dim_;
  int dtype_;
  indexflat_handle_t handle_ = nullptr;
};

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("DTYPE_F16") = py::int_(static_cast<int>(INDEXFLAT_DTYPE_F16));
  m.attr("DTYPE_BF16") = py::int_(static_cast<int>(INDEXFLAT_DTYPE_BF16));
  m.attr("DTYPE_F32") = py::int_(static_cast<int>(INDEXFLAT_DTYPE_F32));
  m.attr("BACKEND_AUTO") = py::int_(static_cast<int>(INDEXFLAT_BACKEND_AUTO));
  m.attr("BACKEND_CPU") = py::int_(static_cast<int>(INDEXFLAT_BACKEND_CPU));
  m.attr("BACKEND_CUDA") = py::int_(static_cast<int>(INDEXFLAT_BACKEND_CUDA));
  m.attr("BACKEND_METAL") = py::int_(static_cast<int>(INDEXFLAT_BACKEND_METAL));

  py::class_<TorchIndexFlatL2>(m, "IndexFlatL2")
      .def(py::init<int, int>(), py::arg("dim"),
           py::arg("dtype") = static_cast<int>(INDEXFLAT_DTYPE_F32))
      .def("add", &TorchIndexFlatL2::add, py::arg("x"))
      .def("search", &TorchIndexFlatL2::search, py::arg("q"), py::arg("k"), py::kw_only(),
           py::arg("backend") = static_cast<int>(INDEXFLAT_BACKEND_AUTO), py::arg("block_n") = 256,
           py::arg("block_d") = 64, py::arg("num_threads") = 0)
      .def_property_readonly("dim", &TorchIndexFlatL2::dim)
      .def_property_readonly("ntotal", &TorchIndexFlatL2::ntotal);
}
