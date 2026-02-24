#include "common/backends.h"

#include <stdexcept>

namespace indexflat::backend {

bool metal_available() {
  return false;
}

void metal_search(const std::vector<uint8_t> &, DType, const std::vector<float> &, int,
                  const float *, const std::vector<float> &, int64_t, int, const SearchOptions &,
                  float *, int32_t *) {
  throw std::runtime_error("Metal backend was not compiled");
}

} // namespace indexflat::backend
