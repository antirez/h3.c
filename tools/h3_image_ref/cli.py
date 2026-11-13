"""Reference-dump harness for the MiniMax-H3-Image path of h3.c.

Runs the diffusers MiniMax-H3-Image pieces one model at a time and writes one
safetensors fixture per stage into misc/fixtures/. Every fixture also holds the
scalar settings it was produced with, as small int32 tensors, so a C test needs
nothing else.

Stages: text, block0, step, f32truth, denoise, vae, e2e (or all), and step_large, edit_vae and the edit_* stages (not in all).
`all` loads the text encoder once and the transformer once: one six-step run
records block 0, steps 0 and 3 and the final latents.

The step fixture holds
latents_in_step{0..5} and noise_pred_step{0..5} (plus step0..step5 scalars)
instead of one latents_in / noise_pred: every step of the six-step run, each
fed the bf16 reference's own latents (teacher forcing).
The f32truth stage is not part of `all`: it reads the step and text fixtures and
writes h3_image_dit_step_f32.safetensors, the velocities of all six steps and
the refined text computed in float32 from the same inputs, one transformer
block at a time from memory-mapped weights (about 2 GB of float32 weights at a
time, CPU), so the bf16 reference noise can be told from an implementation
error.
The edit_vae stage (reference-image editing, VAE side only, CPU, float32, loads
only the VAE) writes h3_image_edit_vae.safetensors. For each of three synthetic
reference images, numbered 0, 1, 2 in the tensor names, it runs what
MiniMaxH3ImageEditResizeStep and MiniMaxH3ImageEditVaeEncoderStep run, then
MiniMaxH3ImageEditPrepareLatentsStep's noising, and records:
  reference_u8_N    [3,H,W] uint8   the input image (seeded gradients, sinusoids, blocks and noise)
  resized_u8_N      [3,Hr,Wr] uint8 VaeImageProcessor.resize (PIL Lanczos) to the bound_short_edge size with the
                                    reference_* bounds; the input unchanged when the short edge is within them
  mean_N, logvar_N  [24,Hv/16,Wv/16] F32  tiled-encoder moments of the resized image cropped top-left to
                                    multiples of 16 and normalised with the ImageNet mean/std; logvar is the
                                    raw chunk, before the posterior clamp
  noise_N           [24,Hv/16,Wv/16] F32  the standard-normal tensor posterior.sample(generator) consumed
  latents_N         [24,h,w] F32    (sample -> float16 -> float32 -> crop to whole 2x2 patches - latents_mean) / latents_std
  reference_noise_N [24,h,w] F32    the noise scale_noise mixes in, drawn from the same generator after noise_N
  noised_N          [24,h,w] F32    scale_noise(latents, tensor(0.999), reference_noise): t * x + (1 - t) * noise in float32
  noised_rows_N     [h/2*w/2,96] F32 patchify_video_latents(noised[:, :, None], (1, 2, 2)), component (channel*2 + y)*2 + x
  sizes_N           int32[12]       reference H, W; resized H, W; VAE input H, W; latent h, w before and after the
                                    patch crop; vision H, W (bound_short_edge with the vision_* bounds)
Also resize_down_in_u8 [3,203,317] and resize_down_out_u8 [3,61,100] (a Lanczos downscale), bounds
int32[4] (reference_min, reference_max, vision_min, vision_max short edge), reference_noise_aug F32[1], seed.

The edit_text, edit_text_small, edit_step, edit_f32truth and edit_e2e stages are the reference-image editing fixtures.
They all start from edit_vae's reference 0 (the same seeded 256x256 image) and the prompt
"<Picture 1> the same fox, now in autumn leaves", and write the scalar settings plus reference_count:
  edit_text        vision copy 1024x1024 after EditResizeStep, 1024 vision tokens, the whole text encoder (62 GB,
                   memory-mapped on the CPU): reference_u8, vision_u8, pixel_values [4096,1536] F32, image_grid_thw
                   [1,3], token_ids [N] i32, token_tags [N] u8, mm_token_type_ids [N] i32, position_ids [3,N] i32
                   (get_rope_index), vision_merged [1024,5120] and vision_deepstack [3,1024,5120] BF16 (a hook on
                   the vision tower), prompt_embeds [N,5120] BF16, prompt
  edit_text_small  the same for the 256x256 image fed straight to the processor (64 vision tokens)
  edit_step        six teacher-forced BF16 transformer steps on the small text, the 256x256 reference encoded
                   directly (16x16 latent, R = 64 rows) and a 512x512 canvas (P = 256 rows): reference_latents_0,
                   reference_posterior_noise_0, reference_aug_noise_0, condition_rows [64,96], latents_in_step{i}
                   [256,96] (generated rows only), noise_pred_step{i} [320,96] (reference rows first),
                   timestep_step{i} [2], timestep_indices and adaln_indices and token_tags [S], position_ids [S,3],
                   refined_text, hidden_in/hidden_out/hidden_out_f32 of block 0 [S,5376], temb_step0 [2,2688]
  edit_f32truth    float32 velocities of the generated rows for the edit_step inputs, blocks outer
  edit_e2e         the whole edit with the pipeline's sizing (VAE copy 1024x1024, R = 1024) on a 512x512 canvas, float32
                   steps-outer Euler loop; text from the edit_text fixture; image_u8, latents_final, noises, condition_rows, token_ids,
                   vision_u8 (what the processor patchifies: after smart_resize) and vision_first_u8 (before it)
--plan builds everything those stages need except the 62 GB models (image, resize, processor, presentation, packed
sequence, timestep and AdaLN indices, the 2 GB VAE for edit_step and edit_f32truth), prints every shape with N, R, P
and S and writes nothing.
Run from the repository root in an environment that has diffusers (with the
MiniMax-H3-Image classes), transformers, torch and safetensors.
"""

import argparse

from .common import DEFAULT_CHECKPOINT, DEFAULT_PROMPT, pick_device
from .edit import (
    EDIT_PROMPT,
    EDIT_PROMPT_STAGES,
    stage_edit_e2e,
    stage_edit_f32truth,
    stage_edit_step,
    stage_edit_text,
    stage_edit_text_small,
    stage_edit_vae,
)
from .f32truth import stage_f32truth, stage_step_large
from .t2i import stage_all, stage_block0, stage_denoise, stage_e2e, stage_step, stage_text, stage_vae


STAGES = {
    "text": stage_text,
    "block0": stage_block0,
    "step": stage_step,
    "f32truth": stage_f32truth,
    "denoise": stage_denoise,
    "vae": stage_vae,
    "e2e": stage_e2e,
    "step_large": stage_step_large,
    "edit_vae": stage_edit_vae,
    "edit_text": stage_edit_text,
    "edit_text_small": stage_edit_text_small,
    "edit_step": stage_edit_step,
    "edit_f32truth": stage_edit_f32truth,
    "edit_e2e": stage_edit_e2e,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=[*STAGES, "all"], required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", default="misc/fixtures")
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--prompt", default=None, help=f"default {DEFAULT_PROMPT!r}, {EDIT_PROMPT!r} for the edit stages")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--min-generation-area", type=int, default=1024)
    parser.add_argument("--bf16-reference", action="store_true", help="step_large, edit_e2e: also run the slow CPU bfloat16 pass")
    parser.add_argument(
        "--plan",
        action="store_true",
        help="edit stages except edit_vae: build everything that needs no 62 GB model (image, resize, processor, "
        "presentation, packed sequence, indices), print every shape, N, R, P and S, write nothing",
    )
    args = parser.parse_args()
    if args.plan and args.stage not in EDIT_PROMPT_STAGES:
        parser.error(f"--plan applies to {', '.join(EDIT_PROMPT_STAGES)}")
    if args.prompt is None:
        args.prompt = EDIT_PROMPT if args.stage in EDIT_PROMPT_STAGES else DEFAULT_PROMPT

    device = pick_device(args.device)
    if args.stage == "all":
        stage_all(args, device)
    else:
        STAGES[args.stage](args, device)
