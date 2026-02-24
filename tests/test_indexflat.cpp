#include "indexflat.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

void reference_search(const std::vector<float> &x, const std::vector<float> &q, int64_t n,
                      int64_t qn, int dim, int k, std::vector<float> *out_d,
                      std::vector<int32_t> *out_i) {
  out_d->assign(static_cast<size_t>(qn) * k, std::numeric_limits<float>::infinity());
  out_i->assign(static_cast<size_t>(qn) * k, std::numeric_limits<int32_t>::max());
  for (int64_t qi = 0; qi < qn; ++qi) {
    for (int64_t xi = 0; xi < n; ++xi) {
      float dist = 0.0f;
      for (int d = 0; d < dim; ++d) {
        const float diff = q[qi * dim + d] - x[xi * dim + d];
        dist += diff * diff;
      }
      float *bd = out_d->data() + qi * k;
      int32_t *bi = out_i->data() + qi * k;
      if (dist > bd[k - 1] || (dist == bd[k - 1] && static_cast<int32_t>(xi) >= bi[k - 1]))
        continue;
      int p = k - 1;
      while (p > 0 &&
             (dist < bd[p - 1] || (dist == bd[p - 1] && static_cast<int32_t>(xi) < bi[p - 1]))) {
        bd[p] = bd[p - 1];
        bi[p] = bi[p - 1];
        --p;
      }
      bd[p] = dist;
      bi[p] = static_cast<int32_t>(xi);
    }
  }
}

void check(bool ok, const char *msg) {
  if (!ok)
    throw std::runtime_error(msg);
}

void test_small_random() {
  constexpr int dim = 64;
  constexpr int64_t n = 1024;
  constexpr int64_t qn = 7;
  constexpr int k = 10;
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> x(static_cast<size_t>(n) * dim);
  std::vector<float> q(static_cast<size_t>(qn) * dim);
  for (float &v : x)
    v = dist(rng);
  for (float &v : q)
    v = dist(rng);

  std::vector<float> ref_d;
  std::vector<int32_t> ref_i;
  reference_search(x, q, n, qn, dim, k, &ref_d, &ref_i);

  indexflat::IndexFlatL2 index(dim, indexflat::DType::F32);
  index.add(x.data(), n);

  std::vector<float> out_d(static_cast<size_t>(qn) * k);
  std::vector<int32_t> out_i(static_cast<size_t>(qn) * k);
  index.search(q.data(), qn, k, out_d.data(), out_i.data(), {.backend = indexflat::Backend::CPU});

  for (size_t i = 0; i < out_i.size(); ++i) {
    check(out_i[i] == ref_i[i], "CPU indices mismatch with reference");
    check(std::fabs(out_d[i] - ref_d[i]) < 1e-4f, "CPU distances mismatch with reference");
  }
}

void test_tie_break() {
  constexpr int dim = 4;
  const std::vector<float> x = {
      1, 2, 3, 4, // idx 0
      1, 2, 3, 4, // idx 1
      2, 2, 2, 2  // idx 2
  };
  const std::vector<float> q = {1, 2, 3, 4};
  indexflat::IndexFlatL2 index(dim, indexflat::DType::F32);
  index.add(x.data(), 3);
  float d[2];
  int32_t i[2];
  index.search(q.data(), 1, 2, d, i);
  check(i[0] == 0 && i[1] == 1, "tie-break should prefer smaller index");
}

void test_backend_parity() {
  constexpr int dim = 32;
  constexpr int64_t n = 256;
  constexpr int64_t qn = 5;
  constexpr int k = 8;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> x(static_cast<size_t>(n) * dim);
  std::vector<float> q(static_cast<size_t>(qn) * dim);
  for (float &v : x)
    v = dist(rng);
  for (float &v : q)
    v = dist(rng);

  indexflat::IndexFlatL2 index(dim, indexflat::DType::F32);
  index.add(x.data(), n);

  std::vector<float> cpu_d(static_cast<size_t>(qn) * k);
  std::vector<int32_t> cpu_i(static_cast<size_t>(qn) * k);
  index.search(q.data(), qn, k, cpu_d.data(), cpu_i.data(), {.backend = indexflat::Backend::CPU});

  if (indexflat::is_backend_available(indexflat::Backend::CUDA)) {
    std::vector<float> d(static_cast<size_t>(qn) * k);
    std::vector<int32_t> i(static_cast<size_t>(qn) * k);
    index.search(q.data(), qn, k, d.data(), i.data(), {.backend = indexflat::Backend::CUDA});
    for (size_t t = 0; t < i.size(); ++t) {
      check(i[t] == cpu_i[t], "CUDA index mismatch");
      check(std::fabs(d[t] - cpu_d[t]) < 1e-4f, "CUDA distance mismatch");
    }
  }

  if (indexflat::is_backend_available(indexflat::Backend::Metal)) {
    std::vector<float> d(static_cast<size_t>(qn) * k);
    std::vector<int32_t> i(static_cast<size_t>(qn) * k);
    index.search(q.data(), qn, k, d.data(), i.data(), {.backend = indexflat::Backend::Metal});
    for (size_t t = 0; t < i.size(); ++t) {
      check(i[t] == cpu_i[t], "Metal index mismatch");
      check(std::fabs(d[t] - cpu_d[t]) < 1e-4f, "Metal distance mismatch");
    }
  }
}

} // namespace

int main() {
  try {
    test_small_random();
    test_tie_break();
    test_backend_parity();
    std::cout << "All tests passed.\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Test failure: " << e.what() << "\n";
    return 1;
  }
}
