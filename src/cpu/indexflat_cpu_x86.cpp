#include "cpu/simd.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace indexflat::cpu {

#if defined(__x86_64__) || defined(_M_X64)
static void cpuid(int out[4], int leaf, int subleaf) {
#if defined(_MSC_VER)
  __cpuidex(out, leaf, subleaf);
#else
  __asm__ __volatile__("cpuid"
                       : "=a"(out[0]), "=b"(out[1]), "=c"(out[2]), "=d"(out[3])
                       : "a"(leaf), "c"(subleaf));
#endif
}
#endif

bool supports_avx2() {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)
  int info[4] = {0, 0, 0, 0};
  cpuid(info, 7, 0);
  return (info[1] & (1 << 5)) != 0;
#else
  return false;
#endif
}

bool supports_avx512f() {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX512F__)
  int info[4] = {0, 0, 0, 0};
  cpuid(info, 7, 0);
  return (info[1] & (1 << 16)) != 0;
#else
  return false;
#endif
}

float dot_f32_avx2(const float *a, const float *b, int dim) {
#if defined(__AVX2__)
  __m256 acc = _mm256_setzero_ps();
  int d = 0;
  for (; d + 8 <= dim; d += 8) {
    const __m256 va = _mm256_loadu_ps(a + d);
    const __m256 vb = _mm256_loadu_ps(b + d);
    acc = _mm256_fmadd_ps(va, vb, acc);
  }
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, acc);
  float out = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
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

float dot_f32_avx512(const float *a, const float *b, int dim) {
#if defined(__AVX512F__)
  __m512 acc = _mm512_setzero_ps();
  int d = 0;
  for (; d + 16 <= dim; d += 16) {
    const __m512 va = _mm512_loadu_ps(a + d);
    const __m512 vb = _mm512_loadu_ps(b + d);
    acc = _mm512_fmadd_ps(va, vb, acc);
  }
  float out = _mm512_reduce_add_ps(acc);
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
