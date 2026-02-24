#include "indexflat.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

int arg_or_default(char **argv, int argc, const std::string &key, int def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i])
      return std::atoi(argv[i + 1]);
  }
  return def;
}

std::string arg_or_default(char **argv, int argc, const std::string &key, const std::string &def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i])
      return argv[i + 1];
  }
  return def;
}

indexflat::Backend parse_backend(const std::string &s) {
  if (s == "cpu")
    return indexflat::Backend::CPU;
  if (s == "cuda")
    return indexflat::Backend::CUDA;
  if (s == "metal")
    return indexflat::Backend::Metal;
  return indexflat::Backend::Auto;
}

indexflat::DType parse_dtype(const std::string &s) {
  if (s == "fp16")
    return indexflat::DType::F16;
  if (s == "bf16")
    return indexflat::DType::BF16;
  return indexflat::DType::F32;
}

} // namespace

int main(int argc, char **argv) {
  const int q = arg_or_default(argv, argc, "--q", 128);
  const int n = arg_or_default(argv, argc, "--n", 1'000'000);
  const int d = arg_or_default(argv, argc, "--d", 768);
  const int k = arg_or_default(argv, argc, "--k", 10);
  const int iters = arg_or_default(argv, argc, "--iters", 5);
  const int block_n = arg_or_default(argv, argc, "--block-n", 256);
  const std::string dtype_s = arg_or_default(argv, argc, "--dtype", "fp32");
  const std::string backend_s = arg_or_default(argv, argc, "--backend", "auto");

  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> xb(static_cast<size_t>(n) * d);
  std::vector<float> xq(static_cast<size_t>(q) * d);
  for (float &v : xb)
    v = dist(rng);
  for (float &v : xq)
    v = dist(rng);

  indexflat::IndexFlatL2 index(d, parse_dtype(dtype_s));
  index.add(xb.data(), n);

  std::vector<float> out_d(static_cast<size_t>(q) * k);
  std::vector<int32_t> out_i(static_cast<size_t>(q) * k);
  indexflat::SearchOptions opts;
  opts.backend = parse_backend(backend_s);
  opts.block_n = block_n;

  for (int i = 0; i < 2; ++i)
    index.search(xq.data(), q, k, out_d.data(), out_i.data(), opts);
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
    index.search(xq.data(), q, k, out_d.data(), out_i.data(), opts);
  const auto t1 = std::chrono::high_resolution_clock::now();
  const double sec = std::chrono::duration<double>(t1 - t0).count();
  const double avg_ms = (sec * 1000.0) / iters;
  const double qps = (static_cast<double>(q) * iters) / sec;
  const double flops = 2.0 * static_cast<double>(q) * n * d * iters;
  const double gflops = flops / sec / 1e9;
  const double x_read = static_cast<double>(n) * d * sizeof(float);
  const double q_read = static_cast<double>(q) * d * sizeof(float);
  const double bytes = (x_read + q_read) * iters;
  const double gbps = bytes / sec / 1e9;

  std::cout << "IndexFlatL2 Route-B microbench\n";
  std::cout << "Q=" << q << " N=" << n << " D=" << d << " k=" << k << " dtype=" << dtype_s
            << " backend=" << backend_s << " block_n=" << block_n << "\n";
  std::cout << "avg_latency_ms=" << avg_ms << " qps=" << qps << " gflops=" << gflops
            << " est_gbps=" << gbps << "\n";
  std::cout << "tip: try --n 10000000 --d 768 --k 10 --q 128 and tune --block-n\n";
  return 0;
}
