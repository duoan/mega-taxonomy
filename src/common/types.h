#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "indexflat.hpp"

namespace indexflat::common {

inline uint32_t bit_cast_u32(float v) {
  uint32_t out;
  std::memcpy(&out, &v, sizeof(uint32_t));
  return out;
}

inline float bit_cast_f32(uint32_t v) {
  float out;
  std::memcpy(&out, &v, sizeof(float));
  return out;
}

inline uint16_t float_to_bf16(float v) {
  const uint32_t bits = bit_cast_u32(v);
  const uint32_t lsb = (bits >> 16) & 1U;
  const uint32_t rounding_bias = 0x7FFFU + lsb;
  return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

inline float bf16_to_float(uint16_t v) {
  return bit_cast_f32(static_cast<uint32_t>(v) << 16);
}

inline uint16_t float_to_f16(float value) {
  const uint32_t x = bit_cast_u32(value);
  const uint32_t sign = (x >> 16) & 0x8000U;
  uint32_t mantissa = x & 0x007FFFFFU;
  int exp = static_cast<int>((x >> 23) & 0xFFU) - 127 + 15;

  if (exp <= 0) {
    if (exp < -10)
      return static_cast<uint16_t>(sign);
    mantissa |= 0x00800000U;
    const uint32_t shifted = mantissa >> (1 - exp);
    const uint32_t rounded = shifted + 0x00001000U;
    return static_cast<uint16_t>(sign | (rounded >> 13));
  }
  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00U);
  }
  const uint32_t rounded = mantissa + 0x00001000U;
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (rounded >> 13));
}

inline float f16_to_float(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h & 0x8000U)) << 16;
  uint32_t exp = (h >> 10) & 0x1FU;
  uint32_t mantissa = h & 0x3FFU;

  if (exp == 0) {
    if (mantissa == 0)
      return bit_cast_f32(sign);
    exp = 1;
    while ((mantissa & 0x400U) == 0) {
      mantissa <<= 1;
      --exp;
    }
    mantissa &= 0x3FFU;
    const uint32_t fexp = exp + (127 - 15);
    const uint32_t bits = sign | (fexp << 23) | (mantissa << 13);
    return bit_cast_f32(bits);
  }
  if (exp == 31) {
    const uint32_t bits = sign | 0x7F800000U | (mantissa << 13);
    return bit_cast_f32(bits);
  }
  const uint32_t fexp = exp + (127 - 15);
  const uint32_t bits = sign | (fexp << 23) | (mantissa << 13);
  return bit_cast_f32(bits);
}

inline size_t dtype_size(DType dtype) {
  switch (dtype) {
  case DType::F16:
  case DType::BF16:
    return 2;
  case DType::F32:
    return 4;
  default:
    throw std::runtime_error("unsupported dtype");
  }
}

inline float load_element(const uint8_t *base, DType dtype, int64_t idx) {
  switch (dtype) {
  case DType::F32: {
    float v;
    std::memcpy(&v, base + idx * 4, sizeof(float));
    return v;
  }
  case DType::F16: {
    uint16_t v;
    std::memcpy(&v, base + idx * 2, sizeof(uint16_t));
    return f16_to_float(v);
  }
  case DType::BF16: {
    uint16_t v;
    std::memcpy(&v, base + idx * 2, sizeof(uint16_t));
    return bf16_to_float(v);
  }
  default:
    return std::numeric_limits<float>::quiet_NaN();
  }
}

inline void append_encoded(std::vector<uint8_t> &dst, const void *x, int64_t n, int dim,
                           DType dtype) {
  const int64_t elements = n * static_cast<int64_t>(dim);
  if (elements <= 0)
    return;
  const size_t old_size = dst.size();
  dst.resize(old_size + static_cast<size_t>(elements) * dtype_size(dtype));

  if (dtype == DType::F32) {
    std::memcpy(dst.data() + old_size, x, static_cast<size_t>(elements) * sizeof(float));
    return;
  }

  const float *src = static_cast<const float *>(x);
  uint8_t *out = dst.data() + old_size;
  if (dtype == DType::F16) {
    for (int64_t i = 0; i < elements; ++i) {
      const uint16_t h = float_to_f16(src[i]);
      std::memcpy(out + i * 2, &h, sizeof(uint16_t));
    }
    return;
  }
  for (int64_t i = 0; i < elements; ++i) {
    const uint16_t b = float_to_bf16(src[i]);
    std::memcpy(out + i * 2, &b, sizeof(uint16_t));
  }
}

} // namespace indexflat::common
