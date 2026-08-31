# NVIDIA GB10 profile

Measured on NVIDIA GB10 with CUDA 13.0 and driver 595.84.  Each result is the
median of three complete runs using the released `MiniMax-H3` checkpoint:

```sh
./h3 -d ./MiniMax-H3 \
  -p "A bright red cube rotates on a white background." \
  --width 256 --height 256 --frames 22 --steps 2 --layers 35 \
  --token-reduction --seed 42 --profile -o OUTPUT.mp4
```

| Mode | DiT load | Denoise | Load + denoise | DiT peak |
| --- | ---: | ---: | ---: | ---: |
| Resident BF16 baseline | 96.563 s | 2.715 s | 99.278 s | 27.063 GB |
| `--ssd-streaming` | 51.829 s | 84.798 s | 136.627 s | 1.630 GB |
| `--use-int8-row-fc2` | 96.491 s | 2.714 s | 99.205 s | 27.063 GB |

All three repetitions within each mode produced the same SHA-256, and all
nine files share SHA-256
`dbae1b441cface55bfe86aaabe78d44c0c05746909f7874908dd2cb298d8c5c8`.
Baseline versus SSD also measures SSIM 1.0.

`--ssd-streaming` reduces the DiT peak by 94.0%, at a 37.6% increase in
load-plus-denoise time for this two-step workload.  Its prefetch read 50.962
GiB per run at a median 0.592 GiB/s, leaving a median 79.488 s of unhidden I/O.
It is therefore useful as an exact low-memory mode, not as the GB10 speed
default.

`--use-int8-row-fc2` is a Metal/M5 specialization and is intentionally a no-op
in the CUDA backend.  The measured 0.001 s denoise difference is noise.  No
FP8 path is exposed by the current backend, and adding one without an
independent numeric oracle would violate the correctness gate.  CUDA device
allocation already uses the GB10 unified physical memory; managed-memory
prefetch would add migration policy without reducing the measured resident
footprint.

No new performance switch is enabled by default: none demonstrated a speedup
while preserving the verified output.  Use resident BF16 for speed and
`--ssd-streaming` only when the 27 GB resident DiT peak is unacceptable.

## Video VAE tiled F32 attention (T29-T32)

Nsight profiling of the max-quality decode (1024x576, 107 frames) attributed
90.5% of the video VAE phase to the scalar F32 attention kernel (1007.06 s of
1112.96 s median; 1728 calls x 582.8 ms at sequence 2805, head_dim 64).  A
tiled F32 kernel (`h3_attention_tiled_f32_kernel<64>`, 8 query rows per block,
online softmax, identical F32 recurrence) now serves non-causal batch-1 F32
attention with head_dim 64; the scalar kernel remains the fallback and is
selectable again with `H3_DISABLE_TILED_ATTENTION=1`.

Median of three `h3_vae_bench_quality` runs (latent 32x36x64x24, 107 frames):

| Metric | Scalar F32 | Tiled F32 |
| --- | ---: | ---: |
| Video VAE decode | 1112.960 s | 247.737 s (4.49x) |
| Max-quality render wall | 33:36.80 | 18:56.36 (1.78x) |
| DiT denoise (unchanged) | 830.999 s | 814.958 s |
| Video VAE peak | 10.26 GB | 10.26 GB |

Quality: CPU oracle parity within max_abs 2e-5 on shapes 13x2x64 and
2805x4x64, Compute Sanitizer clean, `VERIFY: PASS all`.  Matched short render
(512x288/22 frames, same binary, only the VAE path toggled) measures SSIM
0.999303 / PSNR 55.45 dB.  The max-quality MP4 measures SSIM 0.984985 /
PSNR 43.94 dB against the pre-optimization T28 artifact; two identical
rerenders are bit-identical (SSIM 1.0), so the delta is the VAE reduction
order amplified by the high-frequency content and H.264, not run noise.
