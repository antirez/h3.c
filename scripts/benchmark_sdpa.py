#!/usr/bin/env python3
"""Benchmark CUDA SDPA backends on the MiniMax-H3 attention shape."""

import argparse
import statistics

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


BACKENDS = (
    ("flash", SDPBackend.FLASH_ATTENTION),
    ("cudnn", SDPBackend.CUDNN_ATTENTION),
    ("efficient", SDPBackend.EFFICIENT_ATTENTION),
)


def run_backend(name, backend, query, key, value, runs):
    torch.cuda.reset_peak_memory_stats()
    with sdpa_kernel(backend):
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value
        )
        torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value
            )
            end.record()
            end.synchronize()
            times.append(begin.elapsed_time(end) / 1000.0)
    peak = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
    print(
        f"backend={name} median_seconds={statistics.median(times):.6f} "
        f"runs={','.join(f'{value:.6f}' for value in times)} "
        f"peak_gib={peak:.3f}"
    )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=18816)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(42)
    shape = (1, args.heads, args.sequence, args.head_dim)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn_like(query) * 0.1
    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"device={torch.cuda.get_device_name()} capability="
        f"{torch.cuda.get_device_capability()} shape={shape} dtype=bf16"
    )

    outputs = {}
    for name, backend in BACKENDS:
        try:
            outputs[name] = run_backend(
                name, backend, query, key, value, args.runs
            )
        except RuntimeError as error:
            print(f"backend={name} unavailable={error}")

    if "flash" in outputs and "cudnn" in outputs:
        difference = (
            outputs["flash"].float() - outputs["cudnn"].float()
        ).abs()
        print(
            f"flash_vs_cudnn max_abs={difference.max().item():.9g} "
            f"mean_abs={difference.mean().item():.9g}"
        )


if __name__ == "__main__":
    main()
