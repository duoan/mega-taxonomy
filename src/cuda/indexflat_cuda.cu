#include "common/backends.h"

#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/types.h"

namespace indexflat::backend {
namespace {

__device__ __forceinline__ void topk_push(float dist, int idx, float *best_vals, int *best_idx,
                                          int k) {
  if (dist > best_vals[k - 1] || (dist == best_vals[k - 1] && idx >= best_idx[k - 1]))
    return;
  int pos = k - 1;
  while (pos > 0 &&
         (dist < best_vals[pos - 1] || (dist == best_vals[pos - 1] && idx < best_idx[pos - 1]))) {
    best_vals[pos] = best_vals[pos - 1];
    best_idx[pos] = best_idx[pos - 1];
    --pos;
  }
  best_vals[pos] = dist;
  best_idx[pos] = idx;
}

__global__ void fused_streaming_topk_f32(const float *x, const float *x_norms, const float *q,
                                         const float *q_norms, int64_t n, int64_t q_count, int dim,
                                         int k, float *out_d, int32_t *out_i) {
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int warps_per_block = blockDim.x / 32;
  const int64_t qid = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_id;
  if (qid >= q_count)
    return;

  constexpr int kMax = 32;
  float best_vals[kMax];
  int best_idx[kMax];
  if (lane == 0) {
    for (int i = 0; i < k; ++i) {
      best_vals[i] = CUDART_INF_F;
      best_idx[i] = INT_MAX;
    }
  }

  const float *qrow = q + qid * static_cast<int64_t>(dim);
  const float qn = q_norms[qid];

  for (int64_t xi = 0; xi < n; ++xi) {
    float partial = 0.0f;
    const float *xrow = x + xi * static_cast<int64_t>(dim);
    for (int d = lane; d < dim; d += 32)
      partial += qrow[d] * xrow[d];
    for (int offset = 16; offset > 0; offset >>= 1)
      partial += __shfl_down_sync(0xFFFFFFFFU, partial, offset);
    if (lane == 0) {
      const float dist = qn + x_norms[xi] - 2.0f * partial;
      topk_push(dist, static_cast<int>(xi), best_vals, best_idx, k);
    }
  }

  if (lane == 0) {
    for (int i = 0; i < k; ++i) {
      out_d[qid * static_cast<int64_t>(k) + i] = best_vals[i];
      out_i[qid * static_cast<int64_t>(k) + i] = best_idx[i];
    }
  }
}

inline void check_cuda(cudaError_t err, const char *what) {
  if (err != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

} // namespace

bool cuda_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

void cuda_search(const std::vector<uint8_t> &x_data, DType dtype, const std::vector<float> &x_norms,
                 int dim, const float *q, const std::vector<float> &q_norms, int64_t q_count, int k,
                 const SearchOptions &, float *out_dist, int32_t *out_idx) {
  if (k > 32)
    throw std::runtime_error("CUDA path currently supports k <= 32");

  const int64_t n = static_cast<int64_t>(x_norms.size());
  std::vector<float> x_f32(static_cast<size_t>(n) * dim);
  if (dtype == DType::F32) {
    std::memcpy(x_f32.data(), x_data.data(), x_f32.size() * sizeof(float));
  } else {
    for (int64_t i = 0; i < n * static_cast<int64_t>(dim); ++i) {
      x_f32[static_cast<size_t>(i)] = common::load_element(x_data.data(), dtype, i);
    }
  }

  float *d_x = nullptr;
  float *d_xn = nullptr;
  float *d_q = nullptr;
  float *d_qn = nullptr;
  float *d_outd = nullptr;
  int32_t *d_outi = nullptr;

  check_cuda(cudaMalloc(&d_x, x_f32.size() * sizeof(float)), "cudaMalloc d_x");
  check_cuda(cudaMalloc(&d_xn, x_norms.size() * sizeof(float)), "cudaMalloc d_xn");
  check_cuda(cudaMalloc(&d_q, static_cast<size_t>(q_count) * dim * sizeof(float)),
             "cudaMalloc d_q");
  check_cuda(cudaMalloc(&d_qn, q_norms.size() * sizeof(float)), "cudaMalloc d_qn");
  check_cuda(cudaMalloc(&d_outd, static_cast<size_t>(q_count) * k * sizeof(float)),
             "cudaMalloc d_outd");
  check_cuda(cudaMalloc(&d_outi, static_cast<size_t>(q_count) * k * sizeof(int32_t)),
             "cudaMalloc d_outi");

  check_cuda(cudaMemcpy(d_x, x_f32.data(), x_f32.size() * sizeof(float), cudaMemcpyHostToDevice),
             "copy x");
  check_cuda(
      cudaMemcpy(d_xn, x_norms.data(), x_norms.size() * sizeof(float), cudaMemcpyHostToDevice),
      "copy xn");
  check_cuda(cudaMemcpy(d_q, q, static_cast<size_t>(q_count) * dim * sizeof(float),
                        cudaMemcpyHostToDevice),
             "copy q");
  check_cuda(
      cudaMemcpy(d_qn, q_norms.data(), q_norms.size() * sizeof(float), cudaMemcpyHostToDevice),
      "copy qn");

  constexpr int threads = 128;
  constexpr int warps = threads / 32;
  const int blocks = static_cast<int>((q_count + warps - 1) / warps);
  fused_streaming_topk_f32<<<blocks, threads>>>(d_x, d_xn, d_q, d_qn, n, q_count, dim, k, d_outd,
                                                d_outi);
  check_cuda(cudaGetLastError(), "kernel launch");
  check_cuda(cudaDeviceSynchronize(), "kernel sync");

  check_cuda(cudaMemcpy(out_dist, d_outd, static_cast<size_t>(q_count) * k * sizeof(float),
                        cudaMemcpyDeviceToHost),
             "copy outd");
  check_cuda(cudaMemcpy(out_idx, d_outi, static_cast<size_t>(q_count) * k * sizeof(int32_t),
                        cudaMemcpyDeviceToHost),
             "copy outi");

  cudaFree(d_x);
  cudaFree(d_xn);
  cudaFree(d_q);
  cudaFree(d_qn);
  cudaFree(d_outd);
  cudaFree(d_outi);
}

} // namespace indexflat::backend
