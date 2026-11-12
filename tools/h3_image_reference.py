#!/usr/bin/env python3
"""Reference-dump harness for the MiniMax-H3-Image path of h3.c.

Runs the diffusers MiniMax-H3-Image pieces one model at a time and writes one
safetensors fixture per stage into misc/fixtures/. Every fixture also holds the
scalar settings it was produced with, as small int32 tensors, so a C test needs
nothing else.

Stages: text, block0, step, denoise, vae, e2e (or all).
`all` loads the text encoder once and the transformer once: one six-step run
records block 0, steps 0 and 3 and the final latents.

Fixture tensor names that differ from spec 5.3: the step fixture holds
latents_in_step0, latents_in_step3, noise_pred_step0 and noise_pred_step3
(plus step0 and step3 scalars) instead of one latents_in / noise_pred.
Run from the repository root in an environment that has diffusers (with the
MiniMax-H3-Image classes), transformers, torch and safetensors.
"""

import argparse
import copy
import gc
import json
import os
import sys

import torch
from diffusers import AutoencoderKLMiniMaxH3Image, MiniMaxH3ImageTransformer2DModel, MiniMaxH3Scheduler
from diffusers.modular_pipelines.minimax_h3_image.before_denoise import (
    build_image_packed_sequence,
    get_image_canvas_size,
    patchify_video_latents,
)
from diffusers.utils.torch_utils import maybe_adjust_dtype_for_device, randn_tensor
from safetensors.torch import load_file, save_file
from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

DEFAULT_CHECKPOINT = "/Users/ronica/projects/h3-image/MiniMax-H3-Image"
PATCH_SIZE = (1, 2, 2)
LATENT_CHANNELS = 24
VAE_RATIO = 16
CANVAS_MULTIPLE = 32
TEXT_LAYER = 50
MAX_TOKENS = 4096
TEXT_TAG = 1
IMAGE_TAG = 0
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
SAMPLE_SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]
STEP_FIXTURE_STEPS = (0, 3)

FIXTURE_NAMES = {
    "text": "h3_image_text_256.safetensors",
    "block0": "h3_image_dit_block0.safetensors",
    "step": "h3_image_dit_step.safetensors",
    "denoise": "h3_image_denoise.safetensors",
    "vae": "h3_image_vae_decode.safetensors",
    "e2e": "h3_image_e2e_512.safetensors",
}


class StopForward(Exception):
    """Raised from a hook to end a transformer forward early."""


def release(device):
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


def pick_device(name):
    if name == "mps" and not torch.backends.mps.is_available():
        print("MPS is not available, falling back to CPU", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(name)


def scalar(value):
    return torch.tensor([value], dtype=torch.int32)


def settings(args, canvas_height, canvas_width):
    return {
        "seed": scalar(args.seed),
        "min_generation_area": scalar(args.min_generation_area),
        "height": scalar(args.height),
        "width": scalar(args.width),
        "canvas_height": scalar(canvas_height),
        "canvas_width": scalar(canvas_width),
        "latent_height": scalar(canvas_height // VAE_RATIO),
        "latent_width": scalar(canvas_width // VAE_RATIO),
    }


def save_fixture(args, stage, tensors):
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, FIXTURE_NAMES[stage])
    save_file({name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}, path)
    sidecar = {name: list(tensor.shape) + [str(tensor.dtype)] for name, tensor in tensors.items()}
    sidecar["prompt_text"] = args.prompt
    with open(path[: -len(".safetensors")] + ".json", "w") as handle:
        json.dump(sidecar, handle, indent=2)
    print(f"wrote {path}")


def load_fixture(args, stage):
    path = os.path.join(args.out_dir, FIXTURE_NAMES[stage])
    if not os.path.exists(path):
        sys.exit(f"missing fixture {path}: run `--stage {stage}` first")
    return load_file(path)


def canvas(args):
    # Python round() is half-to-even, as nearbyint in h3_image_canvas.
    height, width = get_image_canvas_size(args.height, args.width, args.min_generation_area, CANVAS_MULTIPLE)
    return height, width


def noise_latents(args, canvas_height, canvas_width):
    shape = (1, LATENT_CHANNELS, 1, canvas_height // VAE_RATIO, canvas_width // VAE_RATIO)
    generator = torch.Generator("cpu").manual_seed(args.seed)
    return randn_tensor(shape, generator=generator, device=torch.device("cpu"), dtype=torch.float32)


def run_text(args, device):
    path = os.path.join(args.checkpoint, "text_encoder")
    tokenizer = Qwen2TokenizerFast.from_pretrained(os.path.join(args.checkpoint, "tokenizer"))
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(args.checkpoint, "processor"))
    token_ids = tokenizer(args.prompt, add_special_tokens=False)["input_ids"][:MAX_TOKENS]
    if not token_ids:
        sys.exit("empty prompt")
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    # unverified: the 62 GB text encoder fits MPS alone; use --device cpu otherwise
    text_encoder.eval()
    with torch.no_grad():
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        mm_token_type_ids = torch.tensor(
            processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device
        )
        outputs = text_encoder.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        embeds = outputs.hidden_states[TEXT_LAYER].to(torch.bfloat16).cpu()
    del outputs, text_encoder
    release(device)
    return token_ids, embeds


def save_text(args, token_ids, embeds):
    height, width = canvas(args)
    save_fixture(
        args,
        "text",
        {
            "token_ids": torch.tensor(token_ids, dtype=torch.int32),
            "prompt": torch.tensor(list(args.prompt.encode("utf-8")), dtype=torch.uint8),
            "prompt_embeds": embeds[0],
            **settings(args, height, width),
        },
    )


def stage_text(args, device):
    save_text(args, *run_text(args, device))


def packed_layout(text_tokens, latent_height, latent_width, device):
    text_token_tags = torch.full((text_tokens,), TEXT_TAG, dtype=torch.long)
    position_ids, token_tags, image_indices, text_indices, _ = build_image_packed_sequence(
        text_token_tags, [], latent_height, latent_width, PATCH_SIZE, IMAGE_TAG
    )
    position_ids = position_ids.to(device, maybe_adjust_dtype_for_device(position_ids.dtype, device))
    return position_ids, token_tags.to(device), image_indices.to(device), text_indices.to(device)


def make_scheduler(args, device):
    scheduler = MiniMaxH3Scheduler.from_pretrained(args.checkpoint, subfolder="scheduler")
    # The sampler rounds every sigma through its float32 timestep 1 - sigma.
    timesteps = torch.tensor([1.0 - sigma for sigma in SAMPLE_SIGMAS[:-1]], dtype=torch.float32)
    scheduler.set_timesteps(sigmas=torch.cat([1.0 - timesteps, torch.zeros(1)]), device=device)
    return scheduler


def run_transformer(args, device, prompt_embeds, last_step, capture_block0=False, stop_after_block0=False):
    """Run the first last_step + 1 denoising steps of the transformer on injected noise.

    Returns a dict with the recorded tensors. last_step is 0-based and at most 5.
    """
    canvas_height, canvas_width = canvas(args)
    latent_height, latent_width = canvas_height // VAE_RATIO, canvas_width // VAE_RATIO
    transformer = MiniMaxH3ImageTransformer2DModel.from_pretrained(
        os.path.join(args.checkpoint, "transformer_diffusers"), torch_dtype=torch.bfloat16
    ).to(device)  # unverified: peak memory of the 62 GB .to(device) on MPS; use --device cpu if it swaps
    transformer.eval()
    scheduler = make_scheduler(args, device)

    text_tokens = prompt_embeds.shape[1]
    position_ids, token_tags, image_indices, text_indices = packed_layout(
        text_tokens, latent_height, latent_width, device
    )
    sequence_length = position_ids.shape[0]
    prompt_embeds = prompt_embeds.to(device)
    latents = patchify_video_latents(noise_latents(args, canvas_height, canvas_width), PATCH_SIZE).to(device)

    recorded = {}

    def block0_hook(module, hook_args, hook_kwargs, output):
        if "hidden_in" in recorded:
            return
        hidden_in, temb, adaln_indices = hook_args[0], hook_args[1], hook_args[2]
        recorded["hidden_in"] = hidden_in[0].detach().cpu()
        recorded["temb"] = temb.detach().float().cpu()
        recorded["adaln_indices"] = adaln_indices.detach().to(torch.int32).cpu()
        recorded["hidden_out"] = output[0].detach().cpu()
        reference = copy.deepcopy(module).float().cpu()
        reference._forward_hooks.clear()
        rotary = tuple(part.detach().float().cpu() for part in hook_args[3])
        recorded["hidden_out_f32"] = reference(
            hidden_in.detach().float().cpu(), recorded["temb"], adaln_indices.detach().long().cpu(), rotary
        )[0].detach()
        del reference
        if stop_after_block0:
            raise StopForward

    def refiner_hook(module, hook_args, output):
        if "refined_text" in recorded:
            return
        recorded["refined_text"] = output[0].detach().cpu()

    handles = [transformer.token_refiner.register_forward_hook(refiner_hook)]
    if capture_block0:
        handles.append(transformer.transformer_blocks[0].register_forward_hook(block0_hook, with_kwargs=True))

    recorded["position_ids"] = position_ids.detach().float().cpu()
    try:
        with torch.no_grad():
            for step in range(last_step + 1):
                timestep = scheduler.timesteps[step]
                row_timesteps = torch.full((sequence_length,), float(timestep), dtype=torch.float32)
                unique_timesteps, timestep_indices = torch.unique(row_timesteps, sorted=True, return_inverse=True)
                if step in STEP_FIXTURE_STEPS:
                    recorded[f"latents_in_step{step}"] = latents.detach().cpu()
                try:
                    noise_pred = transformer(
                        hidden_states=latents[None],
                        encoder_hidden_states=prompt_embeds,
                        timestep=unique_timesteps.to(device),
                        timestep_indices=timestep_indices.to(device),
                        pdd_step_index=step,
                        return_dict=False,
                        token_tags=token_tags,
                        position_ids=position_ids,
                        image_indices=image_indices,
                        text_indices=text_indices,
                    )[0]
                except StopForward:
                    break
                if step in STEP_FIXTURE_STEPS:
                    recorded[f"noise_pred_step{step}"] = noise_pred[0].detach().float().cpu()
                latents = scheduler.step(noise_pred[0].float(), timestep, latents, return_dict=False)[0]
    finally:
        for handle in handles:
            handle.remove()
    recorded["latents_final"] = latents.detach().cpu()
    recorded["latent_height"] = latent_height
    recorded["latent_width"] = latent_width
    del transformer, scheduler
    release(device)
    return recorded


def unpatchify(rows, latent_height, latent_width):
    patch_h, patch_w = PATCH_SIZE[1], PATCH_SIZE[2]
    rows = rows.reshape(-1, latent_height // patch_h, latent_width // patch_w, LATENT_CHANNELS, patch_h, patch_w)
    rows = rows.permute(0, 3, 1, 4, 2, 5)
    return rows.reshape(-1, LATENT_CHANNELS, latent_height, latent_width).contiguous()


def prompt_embeds_from_fixture(args):
    return load_fixture(args, "text")["prompt_embeds"][None]


def save_block0(args, recorded):
    height, width = canvas(args)
    save_fixture(
        args,
        "block0",
        {
            "hidden_in": recorded["hidden_in"],
            "temb": recorded["temb"],
            "adaln_indices": recorded["adaln_indices"],
            "position_ids": recorded["position_ids"],
            "hidden_out": recorded["hidden_out"],
            "hidden_out_f32": recorded["hidden_out_f32"],
            "step": scalar(0),
            **settings(args, height, width),
        },
    )


def save_step(args, recorded):
    height, width = canvas(args)
    tensors = {"refined_text": recorded["refined_text"], **settings(args, height, width)}
    for step in STEP_FIXTURE_STEPS:
        tensors[f"latents_in_step{step}"] = recorded[f"latents_in_step{step}"]
        tensors[f"noise_pred_step{step}"] = recorded[f"noise_pred_step{step}"]
        tensors[f"step{step}"] = scalar(step)
    save_fixture(args, "step", tensors)


def final_latents(args, recorded):
    """Noise and denoised latents, both (24, Hl, Wl)."""
    canvas_height, canvas_width = canvas(args)
    noise = noise_latents(args, canvas_height, canvas_width)[0, :, 0]
    final = unpatchify(recorded["latents_final"], recorded["latent_height"], recorded["latent_width"])[0]
    return noise, final


def save_denoise(args, noise, final):
    height, width = canvas(args)
    save_fixture(args, "denoise", {"noise": noise, "latents_final": final, **settings(args, height, width)})


def stage_block0(args, device):
    save_block0(
        args,
        run_transformer(
            args, device, prompt_embeds_from_fixture(args), last_step=0, capture_block0=True, stop_after_block0=True
        ),
    )


def stage_step(args, device):
    save_step(args, run_transformer(args, device, prompt_embeds_from_fixture(args), last_step=max(STEP_FIXTURE_STEPS)))


def stage_denoise(args, device):
    recorded = run_transformer(args, device, prompt_embeds_from_fixture(args), last_step=5)
    save_denoise(args, *final_latents(args, recorded))


def decode_latents(vae, latents, device):
    """Normalised latents (1, 24, H, W) to the un-normalised image in [0, 1], before 8-bit quantisation."""
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1)
    raw = vae.decode(latents.to(device) * std + mean, return_dict=False)[0]
    pixel_mean = torch.tensor(PIXEL_MEAN, device=device).view(1, -1, 1, 1)
    pixel_std = torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1)
    return raw.float().cpu(), (raw.float() * pixel_std + pixel_mean).clamp(0, 1).cpu()


def to_uint8(image):
    # The released pipeline truncates; it does not round.
    return (image * 255).clamp(0, 255).to(torch.uint8)


def load_vae(args, device):
    return AutoencoderKLMiniMaxH3Image.from_pretrained(
        os.path.join(args.checkpoint, "vae_diffusers"), torch_dtype=torch.float32
    ).to(device).eval()


NONSQUARE_LATENT = (10, 37)


def nonsquare_latents(args):
    shape = (1, LATENT_CHANNELS, *NONSQUARE_LATENT)
    generator = torch.Generator("cpu").manual_seed(args.seed)
    return randn_tensor(shape, generator=generator, device=torch.device("cpu"), dtype=torch.float32)


def run_vae(args, latents):
    # The VAE reference is F32 on the CPU, whatever device the other stages use.
    device = torch.device("cpu")
    vae = load_vae(args, device)
    with torch.no_grad():
        _, image = decode_latents(vae, latents, device)
        crop = latents[..., :16, :16].contiguous()
        raw_1, image_1 = decode_latents(vae, crop, device)
        nonsquare = nonsquare_latents(args)
        _, image_nonsquare = decode_latents(vae, nonsquare, device)
    del vae
    release(device)
    return image, crop, raw_1, image_1, nonsquare, image_nonsquare


def vae_latents(args):
    path = os.path.join(args.out_dir, FIXTURE_NAMES["denoise"])
    if os.path.exists(path):
        return load_file(path)["latents_final"][None]
    print("no denoise fixture: decoding seeded noise latents instead", file=sys.stderr)
    height, width = canvas(args)
    return noise_latents(args, height, width)[:, :, 0]


def stage_vae(args, _device):
    latents = vae_latents(args)
    image, crop, raw_1, image_1, nonsquare, image_nonsquare = run_vae(args, latents)
    height, width = canvas(args)
    save_fixture(
        args,
        "vae",
        {
            "latents": latents[0],
            "image_u8": to_uint8(image)[0],
            "latents_1": crop[0],
            "image_1": image_1[0],
            "image_1_raw": raw_1[0],
            "latents_nonsquare": nonsquare[0],
            "image_nonsquare_u8": to_uint8(image_nonsquare)[0],
            **settings(args, height, width),
        },
    )


def save_e2e(args, token_ids, noise, final):
    image = run_vae(args, final[None])[0]
    height, width = canvas(args)
    save_fixture(
        args,
        "e2e",
        {
            "image_u8": to_uint8(image)[0],
            "noise": noise,
            "token_ids": torch.tensor(token_ids, dtype=torch.int32),
            "prompt": torch.tensor(list(args.prompt.encode("utf-8")), dtype=torch.uint8),
            **settings(args, height, width),
        },
    )


def stage_e2e(args, device):
    """Text, denoise and VAE decode in sequence, one model loaded at a time.

    It composes the same computations the separate stages run, so it checks the
    C pipeline end to end but not the harness's own composition of the stages.
    """
    token_ids, embeds = run_text(args, device)
    recorded = run_transformer(args, device, embeds, last_step=5)
    save_e2e(args, token_ids, *final_latents(args, recorded))


def stage_all(args, device):
    token_ids, embeds = run_text(args, device)
    save_text(args, token_ids, embeds)
    recorded = run_transformer(args, device, embeds, last_step=5, capture_block0=True)
    save_block0(args, recorded)
    save_step(args, recorded)
    noise, final = final_latents(args, recorded)
    save_denoise(args, noise, final)
    stage_vae(args, device)
    save_e2e(args, token_ids, noise, final)


STAGES = {
    "text": stage_text,
    "block0": stage_block0,
    "step": stage_step,
    "denoise": stage_denoise,
    "vae": stage_vae,
    "e2e": stage_e2e,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=[*STAGES, "all"], required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", default="misc/fixtures")
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--prompt", default="a red fox in snow")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--min-generation-area", type=int, default=1024)
    args = parser.parse_args()

    device = pick_device(args.device)
    if args.stage == "all":
        stage_all(args, device)
    else:
        STAGES[args.stage](args, device)


if __name__ == "__main__":
    main()
