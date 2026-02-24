#include "indexflat.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include "common/backends.h"
#include "common/norms.h"
#include "common/types.h"

namespace indexflat {

class IndexFlatL2::Impl {
public:
  explicit Impl(int dim, DType dtype) : dim_(dim), dtype_(dtype) {
    if (dim_ <= 0)
      throw std::invalid_argument("dim must be > 0");
  }

  int dim_ = 0;
  DType dtype_ = DType::F32;
  int64_t n_ = 0;
  std::vector<uint8_t> x_data_;
  std::vector<float> x_norms_;
};

namespace {

Backend resolve_backend(Backend requested) {
  if (requested == Backend::CPU)
    return Backend::CPU;
  if (requested == Backend::CUDA) {
    if (!backend::cuda_available())
      throw std::runtime_error("CUDA backend not available");
    return Backend::CUDA;
  }
  if (requested == Backend::Metal) {
    if (!backend::metal_available())
      throw std::runtime_error("Metal backend not available");
    return Backend::Metal;
  }
  if (backend::cuda_available())
    return Backend::CUDA;
  if (backend::metal_available())
    return Backend::Metal;
  return Backend::CPU;
}

} // namespace

IndexFlatL2::IndexFlatL2(int dim, DType dtype) : impl_(new Impl(dim, dtype)) {}
IndexFlatL2::~IndexFlatL2() = default;
IndexFlatL2::IndexFlatL2(IndexFlatL2 &&) noexcept = default;
IndexFlatL2 &IndexFlatL2::operator=(IndexFlatL2 &&) noexcept = default;

void IndexFlatL2::add(const void *x, int64_t n) {
  if (n < 0)
    throw std::invalid_argument("n must be >= 0");
  if (n == 0)
    return;
  if (!x)
    throw std::invalid_argument("x must be non-null");

  const int64_t old_n = impl_->n_;
  common::append_encoded(impl_->x_data_, x, n, impl_->dim_, impl_->dtype_);
  impl_->n_ += n;

  impl_->x_norms_.resize(static_cast<size_t>(impl_->n_));
  const uint8_t *ptr = impl_->x_data_.data();
  for (int64_t i = old_n; i < impl_->n_; ++i) {
    impl_->x_norms_[static_cast<size_t>(i)] = common::l2_sq_row(ptr, impl_->dtype_, i, impl_->dim_);
  }
}

void IndexFlatL2::search(const float *q, int64_t q_count, int k, float *out_distances,
                         int32_t *out_indices, const SearchOptions &options) const {
  if (!q || !out_distances || !out_indices)
    throw std::invalid_argument("null pointer");
  if (q_count < 0)
    throw std::invalid_argument("q_count must be >= 0");
  if (k <= 0)
    throw std::invalid_argument("k must be > 0");
  if (impl_->n_ == 0)
    throw std::runtime_error("index is empty");
  if (k > impl_->n_)
    throw std::invalid_argument("k must be <= ntotal");

  const std::vector<float> q_norms = common::compute_q_norms(q, q_count, impl_->dim_);
  const Backend selected = resolve_backend(options.backend);

  switch (selected) {
  case Backend::CPU:
    backend::cpu_search(impl_->x_data_, impl_->dtype_, impl_->x_norms_, impl_->dim_, q, q_norms,
                        q_count, k, options, out_distances, out_indices);
    return;
  case Backend::CUDA:
    backend::cuda_search(impl_->x_data_, impl_->dtype_, impl_->x_norms_, impl_->dim_, q, q_norms,
                         q_count, k, options, out_distances, out_indices);
    return;
  case Backend::Metal:
    backend::metal_search(impl_->x_data_, impl_->dtype_, impl_->x_norms_, impl_->dim_, q, q_norms,
                          q_count, k, options, out_distances, out_indices);
    return;
  case Backend::Auto:
    break;
  }
  throw std::runtime_error("unreachable backend");
}

int IndexFlatL2::dim() const {
  return impl_->dim_;
}
int64_t IndexFlatL2::ntotal() const {
  return impl_->n_;
}
DType IndexFlatL2::dtype() const {
  return impl_->dtype_;
}

bool is_backend_available(Backend backend) {
  switch (backend) {
  case Backend::CPU:
    return true;
  case Backend::CUDA:
    return backend::cuda_available();
  case Backend::Metal:
    return backend::metal_available();
  case Backend::Auto:
    return true;
  }
  return false;
}

} // namespace indexflat
