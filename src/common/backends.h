#pragma once

#include <cstdint>
#include <vector>

#include "indexflat.hpp"

namespace indexflat::backend {

void cpu_search(const std::vector<uint8_t> &x_data, DType dtype, const std::vector<float> &x_norms,
                int dim, const float *q, const std::vector<float> &q_norms, int64_t q_count, int k,
                const SearchOptions &options, float *out_dist, int32_t *out_idx);

bool cuda_available();
void cuda_search(const std::vector<uint8_t> &x_data, DType dtype, const std::vector<float> &x_norms,
                 int dim, const float *q, const std::vector<float> &q_norms, int64_t q_count, int k,
                 const SearchOptions &options, float *out_dist, int32_t *out_idx);

bool metal_available();
void metal_search(const std::vector<uint8_t> &x_data, DType dtype,
                  const std::vector<float> &x_norms, int dim, const float *q,
                  const std::vector<float> &q_norms, int64_t q_count, int k,
                  const SearchOptions &options, float *out_dist, int32_t *out_idx);

} // namespace indexflat::backend
