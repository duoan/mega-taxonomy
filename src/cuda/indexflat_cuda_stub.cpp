#include "common/backends.h"

#include <stdexcept>

namespace indexflat::backend {

bool cuda_available() {
  return false;
}

void cuda_search(const std::vector<uint8_t> &, DType, const std::vector<float> &, int,
                 const float *, const std::vector<float> &, int64_t, int, const SearchOptions &,
                 float *, int32_t *) {
  throw std::runtime_error("CUDA backend was not compiled");
}

} // namespace indexflat::backend
