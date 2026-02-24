#include <metal_stdlib>
using namespace metal;

kernel void fused_streaming_topk_f32(
    const device float *x [[buffer(0)]], const device float *x_norms [[buffer(1)]],
    const device float *q [[buffer(2)]], const device float *q_norms [[buffer(3)]],
    device float *out_d [[buffer(4)]], device int *out_i [[buffer(5)]],
    constant uint &n [[buffer(6)]], constant uint &dim [[buffer(7)]],
    constant uint &k [[buffer(8)]], uint tid [[thread_position_in_grid]]) {
  if (k == 0 || k > 32)
    return;
  const uint qid = tid;

  float best_vals[32];
  int best_idx[32];
  for (uint i = 0; i < k; ++i) {
    best_vals[i] = INFINITY;
    best_idx[i] = INT_MAX;
  }

  const device float *qrow = q + static_cast<size_t>(qid) * dim;
  const float qn = q_norms[qid];
  for (uint xi = 0; xi < n; ++xi) {
    const device float *xrow = x + static_cast<size_t>(xi) * dim;
    float dot = 0.0f;
    for (uint d = 0; d < dim; ++d)
      dot += qrow[d] * xrow[d];
    const float dist = qn + x_norms[xi] - 2.0f * dot;
    if (dist > best_vals[k - 1] ||
        (dist == best_vals[k - 1] && static_cast<int>(xi) >= best_idx[k - 1]))
      continue;
    int pos = static_cast<int>(k) - 1;
    while (pos > 0 && (dist < best_vals[pos - 1] ||
                       (dist == best_vals[pos - 1] && static_cast<int>(xi) < best_idx[pos - 1]))) {
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
