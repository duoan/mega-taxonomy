#include "indexflat_c_api.h"

#include <memory>
#include <string>

#include "indexflat.hpp"

namespace {

thread_local std::string g_last_error;

void set_error(const std::string &msg) {
  g_last_error = msg;
}

indexflat::DType to_dtype(int dtype) {
  switch (dtype) {
  case INDEXFLAT_DTYPE_F16:
    return indexflat::DType::F16;
  case INDEXFLAT_DTYPE_BF16:
    return indexflat::DType::BF16;
  case INDEXFLAT_DTYPE_F32:
    return indexflat::DType::F32;
  default:
    throw std::invalid_argument("invalid dtype");
  }
}

indexflat::Backend to_backend(int backend) {
  switch (backend) {
  case INDEXFLAT_BACKEND_AUTO:
    return indexflat::Backend::Auto;
  case INDEXFLAT_BACKEND_CPU:
    return indexflat::Backend::CPU;
  case INDEXFLAT_BACKEND_CUDA:
    return indexflat::Backend::CUDA;
  case INDEXFLAT_BACKEND_METAL:
    return indexflat::Backend::Metal;
  default:
    throw std::invalid_argument("invalid backend");
  }
}

} // namespace

extern "C" {

indexflat_handle_t indexflat_create(int dim, int dtype) {
  try {
    auto ptr = std::make_unique<indexflat::IndexFlatL2>(dim, to_dtype(dtype));
    g_last_error.clear();
    return reinterpret_cast<indexflat_handle_t>(ptr.release());
  } catch (const std::exception &e) {
    set_error(e.what());
    return nullptr;
  } catch (...) {
    set_error("unknown error");
    return nullptr;
  }
}

void indexflat_destroy(indexflat_handle_t handle) {
  auto *ptr = reinterpret_cast<indexflat::IndexFlatL2 *>(handle);
  delete ptr;
}

int indexflat_add_f32(indexflat_handle_t handle, const float *x, int64_t n) {
  try {
    if (!handle)
      throw std::invalid_argument("null handle");
    auto *ptr = reinterpret_cast<indexflat::IndexFlatL2 *>(handle);
    ptr->add(x, n);
    g_last_error.clear();
    return 0;
  } catch (const std::exception &e) {
    set_error(e.what());
    return -1;
  } catch (...) {
    set_error("unknown error");
    return -1;
  }
}

int indexflat_search_f32(indexflat_handle_t handle, const float *q, int64_t q_count, int k,
                         float *out_distances, int32_t *out_indices, int backend, int block_n,
                         int block_d, int num_threads) {
  try {
    if (!handle)
      throw std::invalid_argument("null handle");
    auto *ptr = reinterpret_cast<indexflat::IndexFlatL2 *>(handle);
    indexflat::SearchOptions options;
    options.backend = to_backend(backend);
    options.block_n = block_n;
    options.block_d = block_d;
    options.num_threads = num_threads;
    ptr->search(q, q_count, k, out_distances, out_indices, options);
    g_last_error.clear();
    return 0;
  } catch (const std::exception &e) {
    set_error(e.what());
    return -1;
  } catch (...) {
    set_error("unknown error");
    return -1;
  }
}

int64_t indexflat_ntotal(indexflat_handle_t handle) {
  try {
    if (!handle)
      throw std::invalid_argument("null handle");
    auto *ptr = reinterpret_cast<indexflat::IndexFlatL2 *>(handle);
    g_last_error.clear();
    return ptr->ntotal();
  } catch (const std::exception &e) {
    set_error(e.what());
    return -1;
  } catch (...) {
    set_error("unknown error");
    return -1;
  }
}

int indexflat_dim(indexflat_handle_t handle) {
  try {
    if (!handle)
      throw std::invalid_argument("null handle");
    auto *ptr = reinterpret_cast<indexflat::IndexFlatL2 *>(handle);
    g_last_error.clear();
    return ptr->dim();
  } catch (const std::exception &e) {
    set_error(e.what());
    return -1;
  } catch (...) {
    set_error("unknown error");
    return -1;
  }
}

const char *indexflat_last_error(void) {
  return g_last_error.c_str();
}

} // extern "C"
