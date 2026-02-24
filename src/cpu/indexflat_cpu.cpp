#include "common/backends.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "common/topk.h"
#include "common/types.h"
#include "cpu/simd.h"

namespace indexflat::backend {
namespace {

inline float dot_scalar(const uint8_t *x_data, DType dtype, int64_t x_row, const float *q,
                        int dim) {
  const int64_t base = x_row * static_cast<int64_t>(dim);
  float acc = 0.0f;
  for (int d = 0; d < dim; ++d)
    acc += common::load_element(x_data, dtype, base + d) * q[d];
  return acc;
}

inline float dot_best_effort_f32(const uint8_t *x_data, int64_t x_row, const float *q, int dim) {
  const float *x = reinterpret_cast<const float *>(x_data) + x_row * static_cast<int64_t>(dim);
#if defined(__aarch64__)
  return cpu::dot_f32_neon(x, q, dim);
#elif defined(__x86_64__) || defined(_M_X64)
  if (cpu::supports_avx512f())
    return cpu::dot_f32_avx512(x, q, dim);
  if (cpu::supports_avx2())
    return cpu::dot_f32_avx2(x, q, dim);
#endif
  float acc = 0.0f;
  for (int d = 0; d < dim; ++d)
    acc += x[d] * q[d];
  return acc;
}

inline void search_query_streaming(const std::vector<uint8_t> &x_data, DType dtype,
                                   const std::vector<float> &x_norms, int dim, const float *q,
                                   float q_norm, int64_t n, int k, int block_n, float *out_d,
                                   int32_t *out_i) {
  const uint8_t *x_ptr = x_data.data();
  if (k <= 32) {
    common::TopKSmall<32> tk(k);
    for (int64_t xb = 0; xb < n; xb += block_n) {
      const int64_t xend = std::min<int64_t>(xb + block_n, n);
      for (int64_t xi = xb; xi < xend; ++xi) {
        const float dot = (dtype == DType::F32) ? dot_best_effort_f32(x_ptr, xi, q, dim)
                                                : dot_scalar(x_ptr, dtype, xi, q, dim);
        const float dist = q_norm + x_norms[static_cast<size_t>(xi)] - 2.0f * dot;
        tk.push(dist, static_cast<int32_t>(xi));
      }
    }
    tk.store(out_d, out_i);
    return;
  }

  common::TopKLarge tk(k);
  for (int64_t xb = 0; xb < n; xb += block_n) {
    const int64_t xend = std::min<int64_t>(xb + block_n, n);
    tk.clear_local();
    for (int64_t xi = xb; xi < xend; ++xi) {
      const float dot = (dtype == DType::F32) ? dot_best_effort_f32(x_ptr, xi, q, dim)
                                              : dot_scalar(x_ptr, dtype, xi, q, dim);
      const float dist = q_norm + x_norms[static_cast<size_t>(xi)] - 2.0f * dot;
      tk.push_local(dist, static_cast<int32_t>(xi));
    }
    tk.merge_local();
  }
  tk.store(out_d, out_i);
}

inline void merge_candidate(float d, int32_t id, float *best_d, int32_t *best_i, int k) {
  if (!common::better_pair(d, id, best_d[k - 1], best_i[k - 1]))
    return;
  int p = k - 1;
  while (p > 0 && common::better_pair(d, id, best_d[p - 1], best_i[p - 1])) {
    best_d[p] = best_d[p - 1];
    best_i[p] = best_i[p - 1];
    --p;
  }
  best_d[p] = d;
  best_i[p] = id;
}

} // namespace

void cpu_search(const std::vector<uint8_t> &x_data, DType dtype, const std::vector<float> &x_norms,
                int dim, const float *q, const std::vector<float> &q_norms, int64_t q_count, int k,
                const SearchOptions &options, float *out_dist, int32_t *out_idx) {
  const int64_t n = static_cast<int64_t>(x_norms.size());
  if (n <= 0)
    throw std::runtime_error("empty index");
  const int block_n = std::max(1, options.block_n);

  int threads = options.num_threads;
#ifdef _OPENMP
  if (threads <= 0)
    threads = omp_get_max_threads();
#else
  threads = 1;
#endif

  const bool chunk_parallel = (q_count <= 8 && n >= 16384 && threads > 1);
  if (!chunk_parallel) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t qi = 0; qi < q_count; ++qi) {
      search_query_streaming(x_data, dtype, x_norms, dim, q + qi * static_cast<int64_t>(dim),
                             q_norms[static_cast<size_t>(qi)], n, k, block_n,
                             out_dist + qi * static_cast<int64_t>(k),
                             out_idx + qi * static_cast<int64_t>(k));
    }
    return;
  }

  std::vector<float> global_d(static_cast<size_t>(q_count) * k,
                              std::numeric_limits<float>::infinity());
  std::vector<int32_t> global_i(static_cast<size_t>(q_count) * k,
                                std::numeric_limits<int32_t>::max());

#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
  {
    std::vector<float> local_d(static_cast<size_t>(q_count) * k,
                               std::numeric_limits<float>::infinity());
    std::vector<int32_t> local_i(static_cast<size_t>(q_count) * k,
                                 std::numeric_limits<int32_t>::max());

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int64_t xb = 0; xb < n; xb += block_n) {
      const int64_t xend = std::min<int64_t>(xb + block_n, n);
      for (int64_t qi = 0; qi < q_count; ++qi) {
        const float *qrow = q + qi * static_cast<int64_t>(dim);
        if (k <= 32) {
          common::TopKSmall<32> tk(k);
          for (int64_t xi = xb; xi < xend; ++xi) {
            const float dot = (dtype == DType::F32)
                                  ? dot_best_effort_f32(x_data.data(), xi, qrow, dim)
                                  : dot_scalar(x_data.data(), dtype, xi, qrow, dim);
            const float dist =
                q_norms[static_cast<size_t>(qi)] + x_norms[static_cast<size_t>(xi)] - 2.0f * dot;
            tk.push(dist, static_cast<int32_t>(xi));
          }
          float chunk_d[32];
          int32_t chunk_i[32];
          tk.store(chunk_d, chunk_i);
          for (int j = 0; j < k; ++j) {
            merge_candidate(chunk_d[j], chunk_i[j], local_d.data() + qi * k,
                            local_i.data() + qi * k, k);
          }
        } else {
          common::TopKLarge tk(k);
          tk.clear_local();
          for (int64_t xi = xb; xi < xend; ++xi) {
            const float dot = (dtype == DType::F32)
                                  ? dot_best_effort_f32(x_data.data(), xi, qrow, dim)
                                  : dot_scalar(x_data.data(), dtype, xi, qrow, dim);
            const float dist =
                q_norms[static_cast<size_t>(qi)] + x_norms[static_cast<size_t>(xi)] - 2.0f * dot;
            tk.push_local(dist, static_cast<int32_t>(xi));
          }
          tk.merge_local();
          std::vector<float> chunk_d(k);
          std::vector<int32_t> chunk_i(k);
          tk.store(chunk_d.data(), chunk_i.data());
          for (int j = 0; j < k; ++j) {
            merge_candidate(chunk_d[j], chunk_i[j], local_d.data() + qi * k,
                            local_i.data() + qi * k, k);
          }
        }
      }
    }

#ifdef _OPENMP
#pragma omp critical
#endif
    {
      for (int64_t qi = 0; qi < q_count; ++qi) {
        for (int j = 0; j < k; ++j) {
          merge_candidate(local_d[static_cast<size_t>(qi) * k + j],
                          local_i[static_cast<size_t>(qi) * k + j],
                          global_d.data() + qi * static_cast<int64_t>(k),
                          global_i.data() + qi * static_cast<int64_t>(k), k);
        }
      }
    }
  }

  std::copy(global_d.begin(), global_d.end(), out_dist);
  std::copy(global_i.begin(), global_i.end(), out_idx);
}

} // namespace indexflat::backend
