#include "cpu/simd.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace indexflat::cpu {

float dot_f32_neon(const float *a, const float *b, int dim) {
#if defined(__aarch64__)
  float32x4_t acc = vdupq_n_f32(0.0f);
  int d = 0;
  for (; d + 4 <= dim; d += 4) {
    const float32x4_t va = vld1q_f32(a + d);
    const float32x4_t vb = vld1q_f32(b + d);
    acc = vfmaq_f32(acc, va, vb);
  }
  float out = vaddvq_f32(acc);
  for (; d < dim; ++d)
    out += a[d] * b[d];
  return out;
#else
  (void)a;
  (void)b;
  (void)dim;
  return 0.0f;
#endif
}

} // namespace indexflat::cpu
