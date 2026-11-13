# Regenerating the image-edit fixtures

The reference-image editing tests compare the C implementation with a Python
reference harness (`tools/h3_image_reference.py`). This page lists, in
dependency order, the commands that produce the fixtures and run the tests
that consume them. Nothing here runs automatically; `make image-edit-fixtures`
only prints a pointer to this page.

## Requirements

- `PY`: a Python with `diffusers` (the MiniMax-H3-Image classes),
  `transformers`, `torch` and `safetensors`.
- `H3_IMAGE_DIR`: the MiniMax-H3-Image model directory.
- `G`: an optional wrapper prefix, for example a memory guard. Leave it empty
  to run the commands directly.

The harness runs on the CPU from memory-mapped weights. Its peak is evictable
file pages plus a few GB. Run any harness command with `--plan` first to print
the tensor shapes without loading the 62 GB model.

Times and peak memory below are estimates from the text-to-image runs, not
measurements.

```sh
G=""   PY="python -I"
H="$PY tools/h3_image_reference.py --device cpu --out-dir misc/fixtures"
```

## Steps

1. `$H --stage edit_vae`
   Only the reference-encoder test needs this fixture, not steps 2 to 8.
   Light, about 1 minute.
2. `$G $H --stage edit_text_small`
   62 GB text encoder, N=84: 5 to 15 minutes, 4 GB anonymous memory plus page
   cache.
3. `$G $H --stage edit_step`
   BF16 DiT, 6 steps, S=404: about 30 minutes, 6 GB anonymous memory plus page
   cache.
4. `$G $H --stage edit_f32truth`
   F32, each block converted once for all six steps: about 5 minutes, 4 GB
   anonymous memory plus page cache.
5. `$G $H --stage edit_text`
   62 GB text encoder, N=1044: 10 to 30 minutes, 6 GB anonymous memory plus
   page cache.
6. `$G $H --stage edit_e2e`
   F32, steps outer, 300 block conversions, S=2324: 25 to 35 minutes, 8 GB
   anonymous memory plus page cache. `--bf16-reference` also writes the BF16
   image (hours; optional).
7. Text test, with 2 vision-tower passes and 2 text passes (10 to 20 minutes,
   about 8 GB with the layer ring, plus page cache):

   ```sh
   make h3_real_image_edit_text_test
   H3_MEASURE=1 $G ./h3_real_image_edit_text_test $H3_IMAGE_DIR \
     misc/fixtures/h3_image_edit_text.safetensors \
     misc/fixtures/h3_image_edit_text_small.safetensors
   ```
8. DiT test, two loads per mode (stream: 5 to 10 minutes and 4 GiB; bf16:
   5 to 10 minutes and 38 GiB; int8: 5 to 10 minutes and 20 GiB):

   ```sh
   make h3_real_image_edit_dit_test
   rm -f h3_image_edit_dit_velocity.bin
   for mode in stream bf16 int8; do
     H3_MEASURE=1 $G ./h3_real_image_edit_dit_test \
       $H3_IMAGE_DIR/transformer_diffusers \
       misc/fixtures/h3_image_edit_step.safetensors \
       misc/fixtures/h3_image_edit_text_small.safetensors \
       $mode h3_image_edit_dit_velocity.bin \
       misc/fixtures/h3_image_edit_step_f32.safetensors
   done
   ```
9. `make image-edit-check`, then the edit end-to-end test on
   `misc/fixtures/h3_image_edit_e2e_512.safetensors`.

## Measure mode and bounds

`H3_MEASURE=1` turns every numeric bound into a report: the test prints PASS
or OVER and continues, so one run measures all bounds. Exact checks still
fail the run.

The bounds in `tests/test_real_image_edit_text.c` and
`tests/test_real_image_edit_dit.c` are 1.4x the values measured against the
fixtures. After regenerating the fixtures, run steps 7 and 8, replace the
bounds with 1.4x the printed values, then run both tests without `H3_MEASURE`
(`make test` does).
