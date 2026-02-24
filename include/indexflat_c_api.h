#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle.
typedef void *indexflat_handle_t;

// Mirrors indexflat::DType in include/indexflat.hpp.
enum indexflat_dtype_t {
  INDEXFLAT_DTYPE_F16 = 0,
  INDEXFLAT_DTYPE_BF16 = 1,
  INDEXFLAT_DTYPE_F32 = 2,
};

// Mirrors indexflat::Backend in include/indexflat.hpp.
enum indexflat_backend_t {
  INDEXFLAT_BACKEND_AUTO = 0,
  INDEXFLAT_BACKEND_CPU = 1,
  INDEXFLAT_BACKEND_CUDA = 2,
  INDEXFLAT_BACKEND_METAL = 3,
};

indexflat_handle_t indexflat_create(int dim, int dtype);
void indexflat_destroy(indexflat_handle_t handle);

int indexflat_add_f32(indexflat_handle_t handle, const float *x, int64_t n);
int indexflat_search_f32(indexflat_handle_t handle, const float *q, int64_t q_count, int k,
                         float *out_distances, int32_t *out_indices, int backend, int block_n,
                         int block_d, int num_threads);

int64_t indexflat_ntotal(indexflat_handle_t handle);
int indexflat_dim(indexflat_handle_t handle);

// Returns last error message from the calling thread.
const char *indexflat_last_error(void);

#ifdef __cplusplus
}
#endif
