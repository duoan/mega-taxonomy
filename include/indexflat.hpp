#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace indexflat {

enum class DType : uint8_t {
  F16 = 0,
  BF16 = 1,
  F32 = 2,
};

enum class Backend : uint8_t {
  Auto = 0,
  CPU = 1,
  CUDA = 2,
  Metal = 3,
};

struct SearchOptions {
  Backend backend = Backend::Auto;
  int block_n = 256;
  int block_d = 64;
  int num_threads = 0; // 0 = runtime default
};

class IndexFlatL2 {
public:
  explicit IndexFlatL2(int dim, DType dtype = DType::F32);
  ~IndexFlatL2();

  IndexFlatL2(IndexFlatL2 &&) noexcept;
  IndexFlatL2 &operator=(IndexFlatL2 &&) noexcept;

  IndexFlatL2(const IndexFlatL2 &) = delete;
  IndexFlatL2 &operator=(const IndexFlatL2 &) = delete;

  // Adds N vectors in row-major [N, D]. Pointer type must match dtype().
  void add(const void *x, int64_t n);

  // Queries are float32 [Q, D]. Outputs are row-major [Q, k].
  void search(const float *q, int64_t q_count, int k, float *out_distances, int32_t *out_indices,
              const SearchOptions &options = {}) const;

  int dim() const;
  int64_t ntotal() const;
  DType dtype() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

bool is_backend_available(Backend backend);

} // namespace indexflat
