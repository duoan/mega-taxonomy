#include "common/backends.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/types.h"

namespace indexflat::backend {
namespace {

static NSString *kKernelSource = @R"(
#include <metal_stdlib>
using namespace metal;
kernel void fused_streaming_topk_f32(
    const device float* x [[buffer(0)]],
    const device float* x_norms [[buffer(1)]],
    const device float* q [[buffer(2)]],
    const device float* q_norms [[buffer(3)]],
    device float* out_d [[buffer(4)]],
    device int* out_i [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    constant uint& dim [[buffer(7)]],
    constant uint& k [[buffer(8)]],
    uint qid [[thread_position_in_grid]]) {
  if (k == 0 || k > 32) return;
  float best_vals[32];
  int best_idx[32];
  for (uint i = 0; i < k; ++i) { best_vals[i] = INFINITY; best_idx[i] = INT_MAX; }
  const device float* qrow = q + static_cast<size_t>(qid) * dim;
  const float qn = q_norms[qid];
  for (uint xi = 0; xi < n; ++xi) {
    const device float* xrow = x + static_cast<size_t>(xi) * dim;
    float dot = 0.0f;
    for (uint d = 0; d < dim; ++d) dot += qrow[d] * xrow[d];
    const float dist = qn + x_norms[xi] - 2.0f * dot;
    if (dist > best_vals[k - 1] || (dist == best_vals[k - 1] && static_cast<int>(xi) >= best_idx[k - 1])) continue;
    int pos = static_cast<int>(k) - 1;
    while (pos > 0 && (dist < best_vals[pos - 1] || (dist == best_vals[pos - 1] && static_cast<int>(xi) < best_idx[pos - 1]))) {
      best_vals[pos] = best_vals[pos - 1];
      best_idx[pos] = best_idx[pos - 1];
      --pos;
    }
    best_vals[pos] = dist;
    best_idx[pos] = static_cast<int>(xi);
  }
  for (uint i = 0; i < k; ++i) {
    out_d[static_cast<size_t>(qid) * k + i] = best_vals[i];
    out_i[static_cast<size_t>(qid) * k + i] = best_idx[i];
  }
}
)";

void check_or_throw(NSError *err, const char *what) {
  if (err != nil) {
    throw std::runtime_error(std::string(what) + ": " + [[err localizedDescription] UTF8String]);
  }
}

} // namespace

bool metal_available() {
  return MTLCreateSystemDefaultDevice() != nil;
}

void metal_search(const std::vector<uint8_t> &x_data, DType dtype,
                  const std::vector<float> &x_norms, int dim, const float *q,
                  const std::vector<float> &q_norms, int64_t q_count, int k,
                  const SearchOptions &options, float *out_dist, int32_t *out_idx) {
  if (k > 32) {
    cpu_search(x_data, dtype, x_norms, dim, q, q_norms, q_count, k, options, out_dist, out_idx);
    return;
  }
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device)
    throw std::runtime_error("no Metal device");

  NSError *err = nil;
  id<MTLLibrary> lib = [device newLibraryWithSource:kKernelSource options:nil error:&err];
  check_or_throw(err, "newLibraryWithSource");
  id<MTLFunction> fn = [lib newFunctionWithName:@"fused_streaming_topk_f32"];
  if (!fn)
    throw std::runtime_error("missing Metal function fused_streaming_topk_f32");
  id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
  check_or_throw(err, "newComputePipelineState");

  const int64_t n = static_cast<int64_t>(x_norms.size());
  std::vector<float> x_f32(static_cast<size_t>(n) * dim);
  if (dtype == DType::F32) {
    std::memcpy(x_f32.data(), x_data.data(), x_f32.size() * sizeof(float));
  } else {
    for (int64_t i = 0; i < n * static_cast<int64_t>(dim); ++i) {
      x_f32[static_cast<size_t>(i)] = common::load_element(x_data.data(), dtype, i);
    }
  }

  id<MTLBuffer> bx = [device newBufferWithBytes:x_f32.data()
                                         length:x_f32.size() * sizeof(float)
                                        options:MTLResourceStorageModeShared];
  id<MTLBuffer> bxn = [device newBufferWithBytes:x_norms.data()
                                          length:x_norms.size() * sizeof(float)
                                         options:MTLResourceStorageModeShared];
  id<MTLBuffer> bq = [device newBufferWithBytes:q
                                         length:static_cast<size_t>(q_count) * dim * sizeof(float)
                                        options:MTLResourceStorageModeShared];
  id<MTLBuffer> bqn = [device newBufferWithBytes:q_norms.data()
                                          length:q_norms.size() * sizeof(float)
                                         options:MTLResourceStorageModeShared];
  id<MTLBuffer> boutd = [device newBufferWithLength:static_cast<size_t>(q_count) * k * sizeof(float)
                                            options:MTLResourceStorageModeShared];
  id<MTLBuffer> bouti =
      [device newBufferWithLength:static_cast<size_t>(q_count) * k * sizeof(int32_t)
                          options:MTLResourceStorageModeShared];

  uint32_t n_u = static_cast<uint32_t>(n);
  uint32_t dim_u = static_cast<uint32_t>(dim);
  uint32_t k_u = static_cast<uint32_t>(k);
  id<MTLBuffer> bn = [device newBufferWithBytes:&n_u
                                         length:sizeof(uint32_t)
                                        options:MTLResourceStorageModeShared];
  id<MTLBuffer> bdim = [device newBufferWithBytes:&dim_u
                                           length:sizeof(uint32_t)
                                          options:MTLResourceStorageModeShared];
  id<MTLBuffer> bk = [device newBufferWithBytes:&k_u
                                         length:sizeof(uint32_t)
                                        options:MTLResourceStorageModeShared];

  id<MTLCommandQueue> queue = [device newCommandQueue];
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pso];
  [enc setBuffer:bx offset:0 atIndex:0];
  [enc setBuffer:bxn offset:0 atIndex:1];
  [enc setBuffer:bq offset:0 atIndex:2];
  [enc setBuffer:bqn offset:0 atIndex:3];
  [enc setBuffer:boutd offset:0 atIndex:4];
  [enc setBuffer:bouti offset:0 atIndex:5];
  [enc setBuffer:bn offset:0 atIndex:6];
  [enc setBuffer:bdim offset:0 atIndex:7];
  [enc setBuffer:bk offset:0 atIndex:8];

  MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(q_count), 1, 1);
  NSUInteger tgs = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 128);
  MTLSize tg = MTLSizeMake(tgs, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:tg];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];

  std::memcpy(out_dist, [boutd contents], static_cast<size_t>(q_count) * k * sizeof(float));
  std::memcpy(out_idx, [bouti contents], static_cast<size_t>(q_count) * k * sizeof(int32_t));
}

} // namespace indexflat::backend
