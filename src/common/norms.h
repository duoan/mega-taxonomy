#pragma once

#include <cstdint>
#include <vector>

#include "types.h"

namespace indexflat::common {

inline float l2_sq_row(const uint8_t *data, DType dtype, int64_t row, int dim) {
  const int64_t base = row * static_cast<int64_t>(dim);
  float acc = 0.0f;
  for (int d = 0; d < dim; ++d) {
    const float v = load_element(data, dtype, base + d);
    acc += v * v;
  }
  return acc;
}

inline std::vector<float> compute_x_norms(const std::vector<uint8_t> &data, DType dtype, int64_t n,
                                          int dim) {
  std::vector<float> out(n, 0.0f);
  const uint8_t *ptr = data.data();
  for (int64_t i = 0; i < n; ++i)
    out[i] = l2_sq_row(ptr, dtype, i, dim);
  return out;
}

inline std::vector<float> compute_q_norms(const float *q, int64_t q_count, int dim) {
  std::vector<float> out(q_count, 0.0f);
  for (int64_t qi = 0; qi < q_count; ++qi) {
    const float *qr = q + qi * static_cast<int64_t>(dim);
    float acc = 0.0f;
    for (int d = 0; d < dim; ++d)
      acc += qr[d] * qr[d];
    out[qi] = acc;
  }
  return out;
}

} // namespace indexflat::common
