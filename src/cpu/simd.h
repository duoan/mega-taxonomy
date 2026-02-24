#pragma once

#include <cstdint>

namespace indexflat::cpu {

bool supports_avx2();
bool supports_avx512f();
float dot_f32_avx2(const float *a, const float *b, int dim);
float dot_f32_avx512(const float *a, const float *b, int dim);
float dot_f32_neon(const float *a, const float *b, int dim);

} // namespace indexflat::cpu
