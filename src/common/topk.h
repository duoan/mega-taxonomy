#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace indexflat::common {

struct Candidate {
  float dist = std::numeric_limits<float>::infinity();
  int32_t idx = std::numeric_limits<int32_t>::max();
};

inline bool better(const Candidate &a, const Candidate &b) {
  if (a.dist < b.dist)
    return true;
  if (a.dist > b.dist)
    return false;
  return a.idx < b.idx;
}

inline bool better_pair(float dist, int32_t idx, float rhs_dist, int32_t rhs_idx) {
  if (dist < rhs_dist)
    return true;
  if (dist > rhs_dist)
    return false;
  return idx < rhs_idx;
}

template <int kMax> class TopKSmall {
public:
  explicit TopKSmall(int k) : k_(k) {
    for (int i = 0; i < k_; ++i) {
      vals_[i] = std::numeric_limits<float>::infinity();
      idx_[i] = std::numeric_limits<int32_t>::max();
    }
  }

  inline void push(float dist, int32_t id) {
    if (!better_pair(dist, id, vals_[k_ - 1], idx_[k_ - 1]))
      return;
    int pos = k_ - 1;
    while (pos > 0 && better_pair(dist, id, vals_[pos - 1], idx_[pos - 1])) {
      vals_[pos] = vals_[pos - 1];
      idx_[pos] = idx_[pos - 1];
      --pos;
    }
    vals_[pos] = dist;
    idx_[pos] = id;
  }

  inline float worst_dist() const {
    return vals_[k_ - 1];
  }
  inline int32_t worst_idx() const {
    return idx_[k_ - 1];
  }

  inline void store(float *out_d, int32_t *out_i) const {
    for (int i = 0; i < k_; ++i) {
      out_d[i] = vals_[i];
      out_i[i] = idx_[i];
    }
  }

private:
  int k_;
  std::array<float, kMax> vals_{};
  std::array<int32_t, kMax> idx_{};
};

class TopKLarge {
public:
  explicit TopKLarge(int k, int local_k = 64) : k_(k), local_k_(std::min(local_k, k)) {
    vals_.assign(k_, std::numeric_limits<float>::infinity());
    idx_.assign(k_, std::numeric_limits<int32_t>::max());
    local_vals_.assign(local_k_, std::numeric_limits<float>::infinity());
    local_idx_.assign(local_k_, std::numeric_limits<int32_t>::max());
  }

  inline void clear_local() {
    std::fill(local_vals_.begin(), local_vals_.end(), std::numeric_limits<float>::infinity());
    std::fill(local_idx_.begin(), local_idx_.end(), std::numeric_limits<int32_t>::max());
  }

  inline void push_local(float dist, int32_t id) {
    if (!better_pair(dist, id, local_vals_.back(), local_idx_.back()))
      return;
    int pos = local_k_ - 1;
    while (pos > 0 && better_pair(dist, id, local_vals_[pos - 1], local_idx_[pos - 1])) {
      local_vals_[pos] = local_vals_[pos - 1];
      local_idx_[pos] = local_idx_[pos - 1];
      --pos;
    }
    local_vals_[pos] = dist;
    local_idx_[pos] = id;
  }

  inline void merge_local() {
    for (int i = 0; i < local_k_; ++i) {
      push_global(local_vals_[i], local_idx_[i]);
    }
  }

  inline void store(float *out_d, int32_t *out_i) const {
    for (int i = 0; i < k_; ++i) {
      out_d[i] = vals_[i];
      out_i[i] = idx_[i];
    }
  }

private:
  inline void push_global(float dist, int32_t id) {
    if (!better_pair(dist, id, vals_.back(), idx_.back()))
      return;
    int pos = k_ - 1;
    while (pos > 0 && better_pair(dist, id, vals_[pos - 1], idx_[pos - 1])) {
      vals_[pos] = vals_[pos - 1];
      idx_[pos] = idx_[pos - 1];
      --pos;
    }
    vals_[pos] = dist;
    idx_[pos] = id;
  }

  int k_;
  int local_k_;
  std::vector<float> vals_;
  std::vector<int32_t> idx_;
  std::vector<float> local_vals_;
  std::vector<int32_t> local_idx_;
};

} // namespace indexflat::common
