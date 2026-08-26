#!/usr/bin/env python3
"""Simulate int8 QK^T attention accuracy against an F32 SDPA reference.

Answers the phase-A quality question of misc/INT8_QK_DESIGN.md without any
Metal code: quantize Q and K per token row to int8 (127/absmax, matching
h3_quantize_bf16_int8_rows), optionally subtract the per-head K channel mean
before quantizing (SageAttention smooth-K; the removed rank-1 term is
constant per query row and cancels in softmax), then compare the final
attention output against the F32 reference. Reports relative L2 per variant
for plain Gaussian tensors and for tensors with strong K channel outliers,
which is the regime smooth-K exists for. Requires only numpy.
"""

import numpy as np


def quantize_rows(matrix):
    """127/absmax per row, matching the repo's row-wise int8 quantizer."""
    maximum = np.max(np.abs(matrix), axis=-1, keepdims=True)
    scale = np.where(maximum > 0, maximum / 127.0, 1.0 / 127.0)
    quantized = np.clip(np.rint(matrix / scale), -127, 127).astype(np.int8)
    return quantized, scale.astype(np.float32)


def sdpa_f32(query, key, value, scale):
    scores = (query.astype(np.float64) @ key.astype(np.float64).T) * scale
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return (weights @ value.astype(np.float64)).astype(np.float32)


def sdpa_int8_qk(query, key, value, scale, smooth_k):
    key_effective = key.astype(np.float64)
    if smooth_k:
        key_effective = key_effective - key_effective.mean(
            axis=0, keepdims=True)
    q_int8, q_scale = quantize_rows(query.astype(np.float64))
    k_int8, k_scale = quantize_rows(key_effective)
    scores = q_int8.astype(np.float64) @ k_int8.astype(np.float64).T
    scores = scores * q_scale * k_scale.T * scale
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return (weights @ value.astype(np.float64)).astype(np.float32)


def relative_l2(candidate, reference):
    return float(np.linalg.norm(candidate - reference) /
                 np.linalg.norm(reference))


def run_case(name, make_qkv, tokens=2048, head_dim=128, heads=8, seed=7):
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(head_dim)
    errors = {False: [], True: []}
    for _ in range(heads):
        query, key, value = make_qkv(rng, tokens, head_dim)
        reference = sdpa_f32(query, key, value, scale)
        for smooth in (False, True):
            candidate = sdpa_int8_qk(query, key, value, scale, smooth)
            errors[smooth].append(relative_l2(candidate, reference))
    print(f"{name:28s} plain int8-QK L2 {np.mean(errors[False]):.2e}   "
          f"+smooth-K L2 {np.mean(errors[True]):.2e}")


def gaussian(rng, tokens, head_dim):
    return (rng.standard_normal((tokens, head_dim)),
            rng.standard_normal((tokens, head_dim)),
            rng.standard_normal((tokens, head_dim)))


def outlier_k(rng, tokens, head_dim):
    query = rng.standard_normal((tokens, head_dim))
    key = rng.standard_normal((tokens, head_dim))
    # Strong shared per-channel bias on a few K channels, the documented
    # failure mode of naive int8 attention on DiT/LLM checkpoints.
    channels = rng.choice(head_dim, size=6, replace=False)
    key[:, channels] += rng.uniform(8.0, 20.0, size=6) * rng.choice(
        [-1.0, 1.0], size=6)
    value = rng.standard_normal((tokens, head_dim))
    return query, key, value


def main():
    print("int8 QK^T attention-output relative L2 vs F32 SDPA "
          "(2048 tokens, d=128, 8 heads)")
    run_case("gaussian Q/K/V", gaussian)
    run_case("K channel outliers", outlier_k)


if __name__ == "__main__":
    main()
