#!/usr/bin/env python3
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
import copy
import gc
import json
import mmap
import os
import struct
import sys
import types

import numpy as np
import torch
from PIL import Image, ImageOps
from diffusers import AutoencoderKLMiniMaxH3Image, MiniMaxH3ImageTransformer2DModel, MiniMaxH3Scheduler
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.transformers.transformer_minimax_h3_image import (
    MiniMaxH3ImageAdaLayerNormOut,
    MiniMaxH3ImageRotaryPosEmbed,
    MiniMaxH3ImageTokenRefiner,
    MiniMaxH3ImageTransformerBlock,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.modular_pipelines.minimax_h3_image.before_encoder import bound_short_edge
from diffusers.modular_pipelines.minimax_h3_image.encoders import get_qwen3vl_prompt_embeds
from diffusers.modular_pipelines.minimax_h3_image.before_denoise import (
    build_image_packed_sequence,
    get_image_canvas_size,
    patchify_video_latents,
)
from diffusers.utils.torch_utils import maybe_adjust_dtype_for_device, randn_tensor
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

DEFAULT_CHECKPOINT = os.environ.get("H3_IMAGE_DIR", "MiniMax-H3-Image")
DEFAULT_PROMPT = "a red fox in snow"
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
STEP_FIXTURE_STEPS = (0, 1, 2, 3, 4, 5)

FIXTURE_NAMES = {
    "text": "h3_image_text_256.safetensors",
    "block0": "h3_image_dit_block0.safetensors",
    "step": "h3_image_dit_step.safetensors",
    "f32truth": "h3_image_dit_step_f32.safetensors",
    "denoise": "h3_image_denoise.safetensors",
    "vae": "h3_image_vae_decode.safetensors",
    "e2e": "h3_image_e2e_512.safetensors",
    "edit_vae": "h3_image_edit_vae.safetensors",
    "edit_text": "h3_image_edit_text.safetensors",
    "edit_text_small": "h3_image_edit_text_small.safetensors",
    "edit_step": "h3_image_edit_step.safetensors",
    "edit_f32truth": "h3_image_edit_step_f32.safetensors",
    "edit_e2e": "h3_image_edit_e2e_512.safetensors",
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


SAFETENSORS_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


def load_transformer_mmap(directory):
    """The transformer with every weight a view of its memory-mapped shard.

    Unlike from_pretrained this keeps the 62 GB in evictable file-backed pages
    instead of anonymous memory; the dtypes are the checkpoint's own (bfloat16
    everywhere except the float32 output heads).
    """
    config = MiniMaxH3ImageTransformer2DModel.load_config(directory)
    with torch.device("meta"):
        model = MiniMaxH3ImageTransformer2DModel.from_config(config)
    with open(os.path.join(directory, "diffusion_pytorch_model.safetensors.index.json")) as handle:
        weight_map = json.load(handle)["weight_map"]
    maps = {}
    state = {}
    for shard in sorted(set(weight_map.values())):
        handle = open(os.path.join(directory, shard), "rb")
        maps[shard] = (handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_COPY))
        mapped = maps[shard][1]
        (header_length,) = struct.unpack("<Q", mapped[:8])
        header = json.loads(mapped[8 : 8 + header_length])
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            begin, end = entry["data_offsets"]
            dtype = SAFETENSORS_DTYPES[entry["dtype"]]
            count = (end - begin) // torch.empty(0, dtype=dtype).element_size()
            state[name] = torch.frombuffer(mapped, dtype=dtype, count=count, offset=8 + header_length + begin).reshape(
                entry["shape"]
            )
    model.load_state_dict(state, strict=True, assign=True)
    model.rope = MiniMaxH3ImageRotaryPosEmbed(rope_freq_dim=config["rope_freq_dim"], rope_theta=config["rope_theta"])
    for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
        if tensor.device.type == "meta":
            raise RuntimeError(f"{name} was not loaded")
    model._h3_mmaps = maps
    return model.eval()


def run_transformer(args, device, prompt_embeds, last_step, capture_block0=False, stop_after_block0=False):
    """Run the first last_step + 1 denoising steps of the transformer on injected noise.

    Returns a dict with the recorded tensors. last_step is 0-based and at most 5.
    """
    canvas_height, canvas_width = canvas(args)
    latent_height, latent_width = canvas_height // VAE_RATIO, canvas_width // VAE_RATIO
    transformer_directory = os.path.join(args.checkpoint, "transformer_diffusers")
    if device.type == "cpu":
        transformer = load_transformer_mmap(transformer_directory)
    else:
        transformer = MiniMaxH3ImageTransformer2DModel.from_pretrained(
            transformer_directory, torch_dtype=torch.bfloat16
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


class WeightMap:
    """Float32 tensors of the transformer checkpoint, read one at a time from memory-mapped shards."""

    def __init__(self, directory):
        self.directory = directory
        with open(os.path.join(directory, "diffusion_pytorch_model.safetensors.index.json")) as handle:
            self.shards = json.load(handle)["weight_map"]
        self.open_files = {}

    def tensor(self, name):
        shard = self.shards[name]
        if shard not in self.open_files:
            self.open_files[shard] = safe_open(os.path.join(self.directory, shard), "pt")
        return self.open_files[shard].get_tensor(name).float()

    def prefixed(self, prefix):
        return {name[len(prefix) :]: self.tensor(name) for name in self.shards if name.startswith(prefix)}


def f32_refined_text(weights, prompt_embeds):
    refiner = MiniMaxH3ImageTokenRefiner(
        hidden_size=5376,
        num_attention_heads=56,
        attention_head_dim=128,
        ffn_dim=14336,
        num_layers=2,
        norm_eps=1e-5,
        qk_norm_eps=1e-5,
        final_norm_eps=1e-5,
    )
    refiner.load_state_dict(weights.prefixed("token_refiner."))
    projected = torch.nn.functional.linear(
        prompt_embeds.float(), weights.tensor("context_embedder.weight"), weights.tensor("context_embedder.bias")
    )
    return refiner(projected[None])[0]


def f32_velocities(weights, prompt_embeds, step_fixture):
    """Float32 velocity of every step in STEP_FIXTURE_STEPS, plus the refined text."""
    latent_height = int(step_fixture["latent_height"])
    latent_width = int(step_fixture["latent_width"])
    cpu = torch.device("cpu")
    text_tokens = prompt_embeds.shape[0]
    position_ids, token_tags, _, _ = packed_layout(text_tokens, latent_height, latent_width, cpu)
    sequence_length = position_ids.shape[0]
    cos, sin = MiniMaxH3ImageRotaryPosEmbed(16, 10000.0)(position_ids.float())
    with torch.no_grad():
        refined = f32_refined_text(weights, prompt_embeds)
        proj_in = torch.nn.Linear(96, 5376)
        proj_in.load_state_dict(weights.prefixed("proj_in."))
        embedder = TimestepEmbedding(256, 5376, out_dim=2688)
        embedder.load_state_dict(weights.prefixed("time_embedder."))
        time_proj = Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)
        hidden, temb = {}, {}
        for step in STEP_FIXTURE_STEPS:
            hidden[step] = torch.cat([refined, proj_in(step_fixture[f"latents_in_step{step}"])], 0)
            timestep = torch.tensor([1.0 - SAMPLE_SIGMAS[step]], dtype=torch.float32)
            temb[step] = embedder(time_proj(timestep))
        block = MiniMaxH3ImageTransformerBlock(
            hidden_size=5376,
            num_attention_heads=56,
            attention_head_dim=128,
            ffn_dim=14336,
            time_embed_dim=2688,
            norm_eps=1e-5,
            qk_norm_eps=1e-5,
        )
        for index in range(50):
            block.load_state_dict(weights.prefixed(f"transformer_blocks.{index}."))
            for step in STEP_FIXTURE_STEPS:
                hidden[step] = block(hidden[step][None], temb[step], token_tags, (cos, sin))[0]
        norm_out = MiniMaxH3ImageAdaLayerNormOut(5376, 2688, 1e-5)
        norm_out.load_state_dict(weights.prefixed("norm_out."))
        velocity = {}
        for step in STEP_FIXTURE_STEPS:
            normed = norm_out(hidden[step][None], temb[step], torch.zeros(sequence_length, dtype=torch.long))[0]
            head = torch.nn.Linear(5376, 96)
            head.load_state_dict(weights.prefixed(f"pdd_proj_out.{step}."))
            velocity[step] = head(normed)[text_tokens:]
    return refined, velocity


def stage_f32truth(args, _device):
    step_fixture = load_fixture(args, "step")
    prompt_embeds = load_fixture(args, "text")["prompt_embeds"]
    weights = WeightMap(os.path.join(args.checkpoint, "transformer_diffusers"))
    refined, velocity = f32_velocities(weights, prompt_embeds, step_fixture)
    tensors = {"refined_text_f32": refined}
    for step in STEP_FIXTURE_STEPS:
        tensors[f"noise_pred_f32_step{step}"] = velocity[step]
        tensors[f"step{step}"] = scalar(step)
        reference = step_fixture[f"noise_pred_step{step}"]
        error = (torch.linalg.norm(reference - velocity[step]) / torch.linalg.norm(velocity[step])).item()
        print(f"step {step}: bf16 reference vs f32 truth rel-L2 {error:.4g}")
    height, width = canvas(args)
    save_fixture(args, "f32truth", {**tensors, **settings(args, height, width)})


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


def large_step_path(args, canvas_height, canvas_width):
    return os.path.join(
        args.out_dir, f"h3_image_dit_step_large_{canvas_height // VAE_RATIO}x{canvas_width // VAE_RATIO}.safetensors"
    )


def blockwise_step0(weights, prompt_embeds, rows, latent_height, latent_width, with_bf16):
    """Step-0 velocity of the full transformer for the rows: float32, and bfloat16 when with_bf16.

    Both runs go one transformer block at a time from the same memory-mapped weights, so the
    memory stays at one block per dtype whatever the sequence length. The bfloat16 run casts
    the way the diffusers model does: float32 proj_in, time embedder and heads, bfloat16 text
    stream, blocks and output norm. It is about ten times slower than float32 on the CPU
    (63 s against 6.5 s per block at 4.4k rows), so it is off by default.
    """
    import time

    cpu = torch.device("cpu")
    text_tokens = prompt_embeds.shape[0]
    position_ids, token_tags, _, _ = packed_layout(text_tokens, latent_height, latent_width, cpu)
    sequence_length = position_ids.shape[0]
    cos, sin = MiniMaxH3ImageRotaryPosEmbed(16, 10000.0)(position_ids.float())
    block_args = dict(
        hidden_size=5376,
        num_attention_heads=56,
        attention_head_dim=128,
        ffn_dim=14336,
        time_embed_dim=2688,
        norm_eps=1e-5,
        qk_norm_eps=1e-5,
    )
    with torch.no_grad():
        refined_f32 = f32_refined_text(weights, prompt_embeds)
        if with_bf16:
            refiner_bf16 = MiniMaxH3ImageTokenRefiner(
                hidden_size=5376,
                num_attention_heads=56,
                attention_head_dim=128,
                ffn_dim=14336,
                num_layers=2,
                norm_eps=1e-5,
                qk_norm_eps=1e-5,
                final_norm_eps=1e-5,
            )
            refiner_bf16.load_state_dict(weights.prefixed("token_refiner."))
            refiner_bf16 = refiner_bf16.bfloat16()
            projected = torch.nn.functional.linear(
                prompt_embeds.bfloat16(),
                weights.tensor("context_embedder.weight").bfloat16(),
                weights.tensor("context_embedder.bias").bfloat16(),
            )
            refined_bf16 = refiner_bf16(projected[None])[0]
            del refiner_bf16
        proj_in = torch.nn.Linear(96, 5376)
        proj_in.load_state_dict(weights.prefixed("proj_in."))
        embedder = TimestepEmbedding(256, 5376, out_dim=2688)
        embedder.load_state_dict(weights.prefixed("time_embedder."))
        time_proj = Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)
        image = proj_in(rows)
        hidden_f32 = torch.cat([refined_f32, image], 0)
        if with_bf16:
            hidden_bf16 = torch.cat([refined_bf16, image.bfloat16()], 0)
        timestep = torch.tensor([1.0 - SAMPLE_SIGMAS[0]], dtype=torch.float32)
        temb = embedder(time_proj(timestep))
        block = MiniMaxH3ImageTransformerBlock(**block_args)
        started = time.time()
        for index in range(50):
            tick = time.time()
            block.float()
            block.load_state_dict(weights.prefixed(f"transformer_blocks.{index}."))
            hidden_f32 = block(hidden_f32[None], temb, token_tags, (cos, sin))[0]
            middle = time.time()
            if with_bf16:
                block.bfloat16()
                hidden_bf16 = block(hidden_bf16[None], temb, token_tags, (cos, sin))[0]
            print(
                f"block {index}: f32 {middle - tick:.1f}s bf16 {time.time() - middle:.1f}s total {time.time() - started:.0f}s",
                flush=True,
            )
        norm_out = MiniMaxH3ImageAdaLayerNormOut(5376, 2688, 1e-5)
        norm_out.load_state_dict(weights.prefixed("norm_out."))
        head = torch.nn.Linear(5376, 96)
        head.load_state_dict(weights.prefixed("pdd_proj_out.0."))
        zeros = torch.zeros(sequence_length, dtype=torch.long)
        velocity_f32 = head(norm_out(hidden_f32[None], temb, zeros)[0])[text_tokens:]
        velocity_bf16 = None
        if with_bf16:
            norm_out.bfloat16()
            velocity_bf16 = head(norm_out(hidden_bf16[None], temb, zeros)[0].float())[text_tokens:]
    return refined_f32, velocity_f32, velocity_bf16


def stage_step_large(args, _device):
    """Step-0 fixture for a big canvas: seeded noise and the float32 truth velocity (CPU)."""
    height, width = canvas(args)
    latent_height, latent_width = height // VAE_RATIO, width // VAE_RATIO
    prompt_embeds = load_fixture(args, "text")["prompt_embeds"]
    rows = patchify_video_latents(noise_latents(args, height, width), PATCH_SIZE)
    rows = rows.reshape(-1, rows.shape[-1]).contiguous()
    weights = WeightMap(os.path.join(args.checkpoint, "transformer_diffusers"))
    refined, velocity_f32, velocity_bf16 = blockwise_step0(
        weights, prompt_embeds, rows, latent_height, latent_width, args.bf16_reference
    )
    tensors = {
        "refined_text_f32": refined,
        "latents_in_step0": rows,
        "noise_pred_f32_step0": velocity_f32,
        "step0": scalar(0),
        **settings(args, height, width),
    }
    if velocity_bf16 is not None:
        tensors["noise_pred_step0"] = velocity_bf16
        error = (torch.linalg.norm(velocity_bf16 - velocity_f32) / torch.linalg.norm(velocity_f32)).item()
        print(f"bf16 reference vs f32 truth rel-L2 {error:.4g}")
    path = large_step_path(args, height, width)
    os.makedirs(args.out_dir, exist_ok=True)
    save_file({name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}, path)
    print(f"wrote {path}")


REFERENCE_BOUNDS = (1024, 2048, 1024, 1024)
REFERENCE_NOISE_AUG = 0.999
EDIT_IMAGE_SIZES = ((256, 256), (250, 370), (1050, 1046))
RESIZE_DOWN_SIZES = ((203, 317), (61, 100))


def synthetic_image(height, width, seed):
    """A uint8 HxWx3 image with gradients, sinusoids, hard-edged blocks and noise, from a seeded generator."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    y, x = y / height, x / width
    channels = []
    for channel in range(3):
        phase = rng.uniform(0, 2 * np.pi, 2)
        frequency = rng.uniform(2, 9, 2)
        plane = 0.5 * x * (channel + 1) / 3 + 0.3 * y + 0.15 * np.sin(2 * np.pi * frequency[0] * x + phase[0])
        plane += 0.1 * np.cos(2 * np.pi * frequency[1] * y + phase[1])
        channels.append(plane)
    image = np.stack(channels, axis=-1)
    for _ in range(6):
        top, left = int(rng.integers(0, height * 3 // 4)), int(rng.integers(0, width * 3 // 4))
        image[top : top + height // 5, left : left + width // 5] += rng.uniform(-0.3, 0.3, 3)
    image += rng.normal(0, 0.02, image.shape)
    return (np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8)


def encode_reference(vae, scheduler, resized, case_seed):
    """The MiniMaxH3ImageEditVaeEncoderStep and MiniMaxH3ImageEditPrepareLatentsStep computations for one reference."""
    _, patch_h, patch_w = PATCH_SIZE
    pixel_mean = torch.tensor(PIXEL_MEAN).view(1, -1, 1, 1)
    pixel_std = torch.tensor(PIXEL_STD).view(1, -1, 1, 1)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1)
    pixels = torch.from_numpy(np.array(resized)).permute(2, 0, 1)[None]
    pixels = pixels[..., : pixels.shape[-2] // VAE_RATIO * VAE_RATIO, : pixels.shape[-1] // VAE_RATIO * VAE_RATIO]
    vae_size = tuple(pixels.shape[-2:])
    pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
    posterior = vae.encode(pixels, return_dict=False)[0]
    moments = posterior.parameters
    mean, logvar = moments[:, :LATENT_CHANNELS], moments[:, LATENT_CHANNELS:]

    generator = torch.Generator("cpu").manual_seed(case_seed)
    state = generator.get_state()
    sample = posterior.sample(generator=generator)
    replay = torch.Generator("cpu")
    replay.set_state(state)
    noise = randn_tensor(mean.shape, generator=replay, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(sample, posterior.mean + posterior.std * noise), "stored noise is not what sample() drew"

    latents = sample.to(torch.float16).float()
    latent_size = tuple(latents.shape[-2:])
    latents = latents[..., : latents.shape[-2] // patch_h * patch_h, : latents.shape[-1] // patch_w * patch_w]
    latents = (latents - latents_mean) / latents_std
    reference_noise = randn_tensor(latents.shape, generator=generator, device=torch.device("cpu"), dtype=torch.float32)
    noised = scheduler.scale_noise(latents, torch.tensor(REFERENCE_NOISE_AUG), reference_noise)
    rows = patchify_video_latents(noised[:, :, None], PATCH_SIZE)
    return {
        "mean": mean[0],
        "logvar": logvar[0],
        "noise": noise[0],
        "latents": latents[0],
        "reference_noise": reference_noise[0],
        "noised": noised[0],
        "noised_rows": rows,
    }, vae_size, latent_size


def stage_edit_vae(args, _device):
    device = torch.device("cpu")
    processor = VaeImageProcessor(vae_scale_factor=VAE_RATIO, do_normalize=False)
    scheduler = MiniMaxH3Scheduler.from_pretrained(args.checkpoint, subfolder="scheduler")
    vae = load_vae(args, device)
    reference_min, reference_max, vision_min, vision_max = REFERENCE_BOUNDS
    tensors = {}
    with torch.no_grad():
        for index, (height, width) in enumerate(EDIT_IMAGE_SIZES):
            array = synthetic_image(height, width, args.seed + index)
            image = Image.fromarray(array)
            size = bound_short_edge(width, height, reference_min, reference_max, CANVAS_MULTIPLE)
            resized = image if image.size == size else processor.resize(image, height=size[1], width=size[0])
            vision = bound_short_edge(width, height, vision_min, vision_max, CANVAS_MULTIPLE)
            encoded, vae_size, latent_size = encode_reference(vae, scheduler, resized, args.seed + index)
            patched = tuple(encoded["latents"].shape[-2:])
            tensors[f"reference_u8_{index}"] = torch.from_numpy(array).permute(2, 0, 1)
            tensors[f"resized_u8_{index}"] = torch.from_numpy(np.array(resized)).permute(2, 0, 1)
            tensors.update({f"{name}_{index}": value for name, value in encoded.items()})
            tensors[f"sizes_{index}"] = torch.tensor(
                [height, width, resized.size[1], resized.size[0], *vae_size, *latent_size, *patched, vision[1], vision[0]],
                dtype=torch.int32,
            )
            print(f"reference {index}: {width}x{height} -> {resized.size[0]}x{resized.size[1]}, latents {patched}", flush=True)
    del vae
    release(device)
    down_in, down_out = RESIZE_DOWN_SIZES
    source = Image.fromarray(synthetic_image(*down_in, args.seed + 100))
    shrunk = processor.resize(source, height=down_out[0], width=down_out[1])
    tensors["resize_down_in_u8"] = torch.from_numpy(np.array(source)).permute(2, 0, 1)
    tensors["resize_down_out_u8"] = torch.from_numpy(np.array(shrunk)).permute(2, 0, 1)
    tensors["bounds"] = torch.tensor(REFERENCE_BOUNDS, dtype=torch.int32)
    tensors["reference_noise_aug"] = torch.tensor([REFERENCE_NOISE_AUG], dtype=torch.float32)
    tensors["seed"] = scalar(args.seed)
    save_fixture(args, "edit_vae", tensors)


PATCH_ROWS = LATENT_CHANNELS * PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2]
EDIT_PROMPT = "<Picture 1> the same fox, now in autumn leaves"
EDIT_REFERENCE_SIZE = EDIT_IMAGE_SIZES[0]
REFERENCE_PROPS = {"min": REFERENCE_BOUNDS[0], "max": REFERENCE_BOUNDS[1]}
VISION_PROPS = {"min": REFERENCE_BOUNDS[2], "max": REFERENCE_BOUNDS[3]}
BLOCK_ARGS = dict(
    hidden_size=5376,
    num_attention_heads=56,
    attention_head_dim=128,
    ffn_dim=14336,
    time_embed_dim=2688,
    norm_eps=1e-5,
    qk_norm_eps=1e-5,
)


def show(name, value):
    if isinstance(value, torch.Tensor):
        print(f"  {name}: {list(value.shape)} {str(value.dtype).replace('torch.', '')}")
    else:
        print(f"  {name}: {value}")


def edit_settings(args, canvas_height, canvas_width, reference_count):
    return {**settings(args, canvas_height, canvas_width), "reference_count": scalar(reference_count)}


def edit_reference_image(args):
    """The synthetic reference of the edit stages: the image of edit_vae's reference 0 (same generator, same seed)."""
    height, width = EDIT_REFERENCE_SIZE
    return Image.fromarray(synthetic_image(height, width, args.seed))


def pil_to_u8(image):
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous()


def edit_resize(image):
    """MiniMaxH3ImageEditResizeStep on one image: the VAE copy and the vision copy."""
    processor = VaeImageProcessor(vae_scale_factor=VAE_RATIO, do_normalize=False)
    image = ImageOps.exif_transpose(image).convert("RGB")
    if min(image.size) < CANVAS_MULTIPLE:
        sys.exit(f"a reference needs at least {CANVAS_MULTIPLE} pixels on each side, got {image.size}")

    def resize(bounds):
        size = bound_short_edge(*image.size, bounds["min"], bounds["max"], CANVAS_MULTIPLE)
        return image if image.size == size else processor.resize(image, height=size[1], width=size[0])

    return resize(REFERENCE_PROPS), resize(VISION_PROPS)


def rope_positions(model, processor, token_ids, mm_token_type_ids, image_grid_thw):
    """Qwen3VLModel.get_rope_index of one presentation, [3, N] int32.

    model is text_encoder.model, or None for --plan: the same method on a stub that has only the one config value
    it reads (spatial_merge_size), so no 62 GB model is needed.
    """
    arguments = (torch.tensor([token_ids]), torch.tensor(mm_token_type_ids))
    if model is None:
        stub = types.SimpleNamespace(
            config=types.SimpleNamespace(
                vision_config=types.SimpleNamespace(spatial_merge_size=processor.image_processor.merge_size)
            )
        )
        stub.get_vision_position_ids = lambda *a, **k: Qwen3VLModel.get_vision_position_ids(stub, *a, **k)
        positions, _ = Qwen3VLModel.get_rope_index(stub, *arguments, image_grid_thw=image_grid_thw)
    else:
        positions, _ = model.get_rope_index(*arguments, image_grid_thw=image_grid_thw)
    return positions[:, 0].to(torch.int32)


def edit_presentation(tokenizer, processor, prompt, image_grid_thw, model=None):
    """MiniMaxH3ImageEditTextEncoderStep's token list and tags, then create_mm_token_type_ids and get_rope_index."""
    merge = processor.image_processor.merge_size**2
    start, pad, end = (
        tokenizer.convert_tokens_to_ids(token) for token in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")
    )
    token_ids, token_tags = [], []
    for index, grid in enumerate(image_grid_thw):
        label_ids = tokenizer(f"Picture {index + 1}: ", add_special_tokens=False)["input_ids"][:MAX_TOKENS]
        vision_ids = [start] + [pad] * (int(grid.prod()) // merge) + [end]
        token_ids += label_ids + vision_ids
        token_tags += [TEXT_TAG] * len(label_ids) + [IMAGE_TAG] * len(vision_ids)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][:MAX_TOKENS]
    if not prompt_ids:
        sys.exit("empty prompt")
    token_ids += prompt_ids
    token_tags += [TEXT_TAG] * len(prompt_ids)
    mm_token_type_ids = processor.create_mm_token_type_ids([token_ids])
    return {
        "token_ids": token_ids,
        "token_tags": token_tags,
        "mm_token_type_ids": torch.tensor(mm_token_type_ids[0], dtype=torch.int32),
        "position_ids": rope_positions(model, processor, token_ids, mm_token_type_ids, image_grid_thw),
    }


def edit_text_inputs(args, small):
    """The reference image, its processor output and the presentation: everything edit_text needs but the model."""
    reference = edit_reference_image(args)
    # edit_text_small hands the 256x256 image straight to the processor (65536 pixels = min_pixels, no resize, 64 vision
    # tokens): a legitimate processor path, but not the sizing the pipeline applies.
    vae_copy, vision_image = (reference, reference) if small else edit_resize(reference)
    tokenizer = Qwen2TokenizerFast.from_pretrained(os.path.join(args.checkpoint, "tokenizer"))
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(args.checkpoint, "processor"))
    vision = processor.image_processor(images=[vision_image], return_tensors="pt")
    return {
        "reference": reference,
        "vae_copy": vae_copy,
        "vision_image": vision_image,
        "tokenizer": tokenizer,
        "processor": processor,
        "pixel_values": vision["pixel_values"],
        "image_grid_thw": vision["image_grid_thw"],
        **edit_presentation(tokenizer, processor, args.prompt, vision["image_grid_thw"]),
    }


def print_text_plan(args, inputs, name):
    print(f"plan {name}: prompt {args.prompt!r}")
    show("reference_u8", pil_to_u8(inputs["reference"]))
    show("vision_u8", pil_to_u8(inputs["vision_image"]))
    show("pixel_values", inputs["pixel_values"])
    show("image_grid_thw", inputs["image_grid_thw"].to(torch.int32))
    show("token_ids", torch.tensor(inputs["token_ids"], dtype=torch.int32))
    show("token_tags", torch.tensor(inputs["token_tags"], dtype=torch.uint8))
    show("mm_token_type_ids", inputs["mm_token_type_ids"])
    show("position_ids", inputs["position_ids"])
    tokens = int(inputs["image_grid_thw"][0].prod()) // inputs["processor"].image_processor.merge_size**2
    show("vision_merged (planned)", torch.empty(tokens, 5120, dtype=torch.bfloat16))
    show("vision_deepstack (planned)", torch.empty(3, tokens, 5120, dtype=torch.bfloat16))
    n = len(inputs["token_ids"])
    show("prompt_embeds (planned)", torch.empty(n, 5120, dtype=torch.bfloat16))
    print(f"  N = {n} text rows, {tokens} vision tokens")


def load_text_encoder(args, device):
    """On the CPU transformers memory-maps the shards, so the 62 GB stay evictable file pages."""
    path = os.path.join(args.checkpoint, "text_encoder")
    return Qwen3VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()


def stage_edit_text_any(args, device, name):
    small = name == "edit_text_small"
    inputs = edit_text_inputs(args, small)
    if args.plan:
        print_text_plan(args, inputs, name)
        return
    processor = inputs["processor"]
    text_encoder = load_text_encoder(args, device)
    positions = edit_presentation(
        inputs["tokenizer"], processor, args.prompt, inputs["image_grid_thw"], model=text_encoder.model
    )["position_ids"]
    assert torch.equal(positions, inputs["position_ids"]), "the stub's get_rope_index differs from the model's"
    captured = {}

    def vision_hook(module, hook_args, output):
        captured.setdefault("merged", output.pooler_output.detach().to(torch.bfloat16).cpu())
        captured.setdefault("deepstack", torch.stack(list(output.deepstack_features)).detach().to(torch.bfloat16).cpu())

    handle = text_encoder.model.visual.register_forward_hook(vision_hook)
    try:
        with torch.no_grad():
            embeds = get_qwen3vl_prompt_embeds(
                text_encoder,
                processor,
                inputs["token_ids"],
                {"pixel_values": inputs["pixel_values"], "image_grid_thw": inputs["image_grid_thw"]},
                text_encoder_layer=TEXT_LAYER,
                device=device,
                dtype=text_encoder.dtype,
            )[0].to(torch.bfloat16).cpu()
    finally:
        handle.remove()
    del text_encoder
    release(device)
    height, width = canvas(args)
    save_fixture(
        args,
        name,
        {
            "reference_u8": pil_to_u8(inputs["reference"]),
            "vision_u8": pil_to_u8(inputs["vision_image"]),
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"].to(torch.int32),
            "token_ids": torch.tensor(inputs["token_ids"], dtype=torch.int32),
            "token_tags": torch.tensor(inputs["token_tags"], dtype=torch.uint8),
            "mm_token_type_ids": inputs["mm_token_type_ids"],
            "position_ids": inputs["position_ids"],
            "vision_merged": captured["merged"],
            "vision_deepstack": captured["deepstack"],
            "prompt_embeds": embeds,
            "prompt": torch.tensor(list(args.prompt.encode("utf-8")), dtype=torch.uint8),
            **edit_settings(args, height, width, 1),
        },
    )


def stage_edit_text(args, device):
    """Vision copy 1024x1024 (1024 tokens) through the whole text encoder; N is about 1043."""
    stage_edit_text_any(args, device, "edit_text")


def stage_edit_text_small(args, device):
    """The same with the 256x256 image fed straight to the processor (64 vision tokens); N is about 83."""
    stage_edit_text_any(args, device, "edit_text_small")


def edit_layout(args, text_tags, reference_grids, canvas_height, canvas_width):
    """build_image_packed_sequence and the per-step row_timestep_plan of MiniMaxH3ImageSetTimestepsStep."""
    latent_height, latent_width = canvas_height // VAE_RATIO, canvas_width // VAE_RATIO
    position_ids, token_tags, image_indices, text_indices, reference_rows = build_image_packed_sequence(
        torch.as_tensor(text_tags, dtype=torch.long), reference_grids, latent_height, latent_width, PATCH_SIZE, IMAGE_TAG
    )
    sequence_length = position_ids.shape[0]
    scheduler = make_scheduler(args, torch.device("cpu"))
    plan = []
    for timestep in scheduler.timesteps:
        row_timesteps = torch.full((sequence_length,), float(timestep), dtype=torch.float32)
        row_timesteps[image_indices[:reference_rows]] = REFERENCE_NOISE_AUG
        unique_timesteps, timestep_indices = torch.unique(row_timesteps, sorted=True, return_inverse=True)
        plan.append((timestep, unique_timesteps, timestep_indices))
    return {
        "latent_height": latent_height,
        "latent_width": latent_width,
        "position_ids": position_ids,
        "token_tags": token_tags,
        "image_indices": image_indices,
        "text_indices": text_indices,
        "reference_rows": reference_rows,
        "target_rows": (latent_height // PATCH_SIZE[1]) * (latent_width // PATCH_SIZE[2]),
        "text_rows": len(text_tags),
        "sequence_length": sequence_length,
        "plan": plan,
        "adaln_indices": plan[0][2] * 3 + token_tags,
    }


def print_layout_plan(layout):
    show("position_ids", layout["position_ids"].float())
    show("token_tags", layout["token_tags"].to(torch.uint8))
    show("timestep_indices", layout["plan"][0][2].to(torch.int32))
    show("adaln_indices", layout["adaln_indices"].to(torch.int32))
    show("timestep_step0", layout["plan"][0][1])
    for step, (_, unique, indices) in enumerate(layout["plan"]):
        assert unique.numel() == 2 and torch.equal(indices, layout["plan"][0][2]), f"step {step}: time rows moved"
    counts = torch.bincount(layout["adaln_indices"], minlength=6).tolist()
    print(f"  adaln_indices counts for values 0..5: {counts} (0 vision and generated rows, 1 text, 3 reference rows)")
    print(
        f"  N = {layout['text_rows']}, R = {layout['reference_rows']}, P = {layout['target_rows']}, "
        f"S = {layout['sequence_length']}, latent {layout['latent_height']}x{layout['latent_width']}"
    )


def edit_inputs(args, kind, step_fixture=None):
    """Everything edit_step, edit_f32truth and edit_e2e feed the transformer, except the large models.

    kind "step": the 256x256 image goes to the VAE directly (16x16 latent, 64 rows) and the text comes from the
    edit_text_small fixture; kind "e2e": the pipeline's sizing (VAE copy 1024x1024, 64x64 latent, 1024 rows) and the
    edit_text fixture. With --plan the text fixture is replaced by a zero embedding of the right shape and the e2e
    reference latents by zero rows; the step plan still runs the 2 GB VAE.
    """
    small = kind == "step"
    text = edit_text_inputs(args, small)
    fixture = "edit_text_small" if small else "edit_text"
    if args.plan:
        prompt_embeds = torch.zeros(len(text["token_ids"]), 5120, dtype=torch.bfloat16)
        text_tags = torch.tensor(text["token_tags"], dtype=torch.long)
    else:
        loaded = load_fixture(args, fixture)
        prompt_embeds = loaded["prompt_embeds"]
        text_tags = loaded["token_tags"].long()
        assert text_tags.tolist() == text["token_tags"], f"{fixture} was made with another prompt or image"
    reference_image = text["vae_copy"]
    result = {"text": text, "prompt_embeds": prompt_embeds, "text_tags": text_tags}
    if step_fixture is not None:
        result["condition_rows"] = step_fixture["condition_rows"]
        grid = tuple(step_fixture["reference_latents_0"].shape[-2:])
    elif small or not args.plan:
        scheduler = MiniMaxH3Scheduler.from_pretrained(args.checkpoint, subfolder="scheduler")
        vae = load_vae(args, torch.device("cpu"))
        with torch.no_grad():
            encoded, _, _ = encode_reference(vae, scheduler, reference_image, args.seed)
        del vae
        release(torch.device("cpu"))
        result["encoded"] = encoded
        result["condition_rows"] = encoded["noised_rows"]
        grid = tuple(encoded["latents"].shape[-2:])
    else:
        width, height = reference_image.size
        grid = (height // VAE_RATIO // 2 * 2, width // VAE_RATIO // 2 * 2)
        result["condition_rows"] = torch.zeros(grid[0] // 2 * (grid[1] // 2), PATCH_ROWS)
    height, width = canvas(args)
    result["grid"] = grid
    result["canvas"] = (height, width)
    result["layout"] = edit_layout(args, text_tags, [grid], height, width)
    return result


def print_inputs_plan(name, inputs):
    print(f"plan {name}: VAE input {inputs['text']['vae_copy'].size[::-1]} -> reference latent grid {inputs['grid']}")
    print(f"  vision tokens {inputs['text']['image_grid_thw'][0].prod().item() // 4}, canvas {inputs['canvas']}")
    show("prompt_embeds", inputs["prompt_embeds"])
    show("condition_rows", inputs["condition_rows"])
    print_layout_plan(inputs["layout"])


def open_transformer(args, device):
    directory = os.path.join(args.checkpoint, "transformer_diffusers")
    if device.type == "cpu":
        return load_transformer_mmap(directory)
    return MiniMaxH3ImageTransformer2DModel.from_pretrained(directory, torch_dtype=torch.bfloat16).to(device).eval()


def run_edit_transformer(args, device, inputs, noise_rows, last_step=5, capture_block0=False):
    """The diffusers transformer (BF16) through the first last_step + 1 steps of an edit on injected noise.

    Reference rows ride in front of the generated rows at t = 0.999 and are never written by the scheduler step.
    """
    layout = inputs["layout"]
    reference_rows = layout["reference_rows"]
    transformer = open_transformer(args, device)
    scheduler = make_scheduler(args, device)
    prompt_embeds = inputs["prompt_embeds"][None].to(device)
    position_ids = layout["position_ids"].to(device, maybe_adjust_dtype_for_device(layout["position_ids"].dtype, device))
    condition_rows = inputs["condition_rows"].to(device)
    latents = noise_rows.to(device)
    recorded = {}

    def block0_hook(module, hook_args, hook_kwargs, output):
        if "hidden_in" in recorded:
            return
        hidden_in, temb, adaln_indices = hook_args[0], hook_args[1], hook_args[2]
        recorded["hidden_in"] = hidden_in[0].detach().cpu()
        recorded["temb"] = temb.detach().float().cpu()
        recorded["hidden_out"] = output[0].detach().cpu()
        reference = copy.deepcopy(module).float().cpu()
        reference._forward_hooks.clear()
        rotary = tuple(part.detach().float().cpu() for part in hook_args[3])
        recorded["hidden_out_f32"] = reference(
            hidden_in.detach().float().cpu(), recorded["temb"], adaln_indices.detach().long().cpu(), rotary
        )[0].detach()
        del reference

    def refiner_hook(module, hook_args, output):
        recorded.setdefault("refined_text", output[0].detach().cpu())

    handles = [transformer.token_refiner.register_forward_hook(refiner_hook)]
    if capture_block0:
        handles.append(transformer.transformer_blocks[0].register_forward_hook(block0_hook, with_kwargs=True))
    try:
        with torch.no_grad():
            for step in range(last_step + 1):
                timestep, unique_timesteps, timestep_indices = layout["plan"][step]
                rows = torch.cat([condition_rows, latents])
                recorded[f"latents_in_step{step}"] = latents.detach().cpu()
                noise_pred = transformer(
                    hidden_states=rows[None],
                    encoder_hidden_states=prompt_embeds,
                    timestep=unique_timesteps.to(device),
                    timestep_indices=timestep_indices.to(device),
                    pdd_step_index=step,
                    return_dict=False,
                    token_tags=layout["token_tags"].to(device),
                    position_ids=position_ids,
                    image_indices=layout["image_indices"].to(device),
                    text_indices=layout["text_indices"].to(device),
                )[0]
                recorded[f"noise_pred_step{step}"] = noise_pred[0].detach().float().cpu()
                recorded[f"timestep_step{step}"] = unique_timesteps.detach().cpu()
                latents = scheduler.step(noise_pred[0, reference_rows:].float(), timestep, latents, return_dict=False)[0]
    finally:
        for handle in handles:
            handle.remove()
    recorded["latents_final"] = latents.detach().cpu()
    del transformer, scheduler
    release(device)
    return recorded


def edit_noise_rows(args, canvas_height, canvas_width):
    return patchify_video_latents(noise_latents(args, canvas_height, canvas_width), PATCH_SIZE).contiguous()


def stage_edit_step(args, device):
    """Six teacher-forced steps of the BF16 transformer on a 256x256 reference (64 rows) and a 512x512 canvas (256 rows).

    latents_in_step{i} are the generated rows only; noise_pred_step{i} holds the reference rows first, as the model
    returns them. temb_step0 is [2, 2688]: the time embedding of [t_step, 0.999].
    """
    inputs = edit_inputs(args, "step")
    layout = inputs["layout"]
    if args.plan:
        print_inputs_plan("edit_step", inputs)
        show("latents_in_stepN (planned)", torch.empty(layout["target_rows"], PATCH_ROWS))
        show("noise_pred_stepN (planned)", torch.empty(layout["reference_rows"] + layout["target_rows"], PATCH_ROWS))
        show("temb_step0 (planned)", torch.empty(2, 2688))
        show("refined_text (planned)", torch.empty(layout["text_rows"], 5376, dtype=torch.bfloat16))
        show("hidden_in/hidden_out (planned)", torch.empty(layout["sequence_length"], 5376, dtype=torch.bfloat16))
        show("hidden_out_f32 (planned)", torch.empty(layout["sequence_length"], 5376))
        return
    recorded = run_edit_transformer(
        args, device, inputs, edit_noise_rows(args, *inputs["canvas"]), last_step=max(STEP_FIXTURE_STEPS), capture_block0=True
    )
    encoded = inputs["encoded"]
    tensors = {
        "reference_latents_0": encoded["latents"],
        "reference_posterior_noise_0": encoded["noise"],
        "reference_aug_noise_0": encoded["reference_noise"],
        "condition_rows": inputs["condition_rows"],
        "refined_text": recorded["refined_text"],
        "timestep_indices": layout["plan"][0][2].to(torch.int32),
        "adaln_indices": layout["adaln_indices"].to(torch.int32),
        "token_tags": layout["token_tags"].to(torch.uint8),
        "position_ids": layout["position_ids"].float(),
        "hidden_in": recorded["hidden_in"],
        "hidden_out": recorded["hidden_out"],
        "hidden_out_f32": recorded["hidden_out_f32"],
        "temb_step0": recorded["temb"],
        **edit_settings(args, *inputs["canvas"], 1),
    }
    for step in STEP_FIXTURE_STEPS:
        tensors[f"latents_in_step{step}"] = recorded[f"latents_in_step{step}"]
        tensors[f"noise_pred_step{step}"] = recorded[f"noise_pred_step{step}"]
        tensors[f"timestep_step{step}"] = recorded[f"timestep_step{step}"]
        tensors[f"step{step}"] = scalar(step)
    save_fixture(args, "edit_step", tensors)


class EditF32:
    """The float32 pieces of the edit transformer that every step shares: refined text, projections, rope, indices."""

    def __init__(self, weights, inputs):
        layout = inputs["layout"]
        self.weights = weights
        self.text_rows = layout["text_rows"]
        self.reference_rows = layout["reference_rows"]
        self.rotary = MiniMaxH3ImageRotaryPosEmbed(16, 10000.0)(layout["position_ids"].float())
        self.adaln_indices = layout["adaln_indices"].long()
        self.timestep_indices = layout["plan"][0][2].long()
        with torch.no_grad():
            self.refined = f32_refined_text(weights, inputs["prompt_embeds"])
            self.proj_in = torch.nn.Linear(96, 5376)
            self.proj_in.load_state_dict(weights.prefixed("proj_in."))
            embedder = TimestepEmbedding(256, 5376, out_dim=2688)
            embedder.load_state_dict(weights.prefixed("time_embedder."))
            time_proj = Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.temb = [embedder(time_proj(unique.float())) for _, unique, _ in layout["plan"]]
        self.condition_rows = inputs["condition_rows"].float()
        self.block = MiniMaxH3ImageTransformerBlock(**BLOCK_ARGS)

    def start(self, generated_rows):
        return torch.cat([self.refined, self.proj_in(torch.cat([self.condition_rows, generated_rows]))], 0)

    def load_block(self, index):
        self.block.load_state_dict(self.weights.prefixed(f"transformer_blocks.{index}."))

    def apply_block(self, hidden, step):
        return self.block(hidden[None], self.temb[step], self.adaln_indices, self.rotary)[0]

    def run_block(self, index, hidden, step):
        self.load_block(index)
        return self.apply_block(hidden, step)

    def velocity(self, hidden, step):
        norm_out = MiniMaxH3ImageAdaLayerNormOut(5376, 2688, 1e-5)
        norm_out.load_state_dict(self.weights.prefixed("norm_out."))
        head = torch.nn.Linear(5376, 96)
        head.load_state_dict(self.weights.prefixed(f"pdd_proj_out.{step}."))
        normed = norm_out(hidden[None], self.temb[step], self.timestep_indices)[0]
        return head(normed)[self.text_rows + self.reference_rows :]


def f32_edit_velocities(weights, inputs, step_latents, blocks=50):
    """Float32 velocity of the generated rows for each teacher-forced input, blocks outer so each block loads once."""
    state = EditF32(weights, inputs)
    with torch.no_grad():
        hidden = {step: state.start(rows) for step, rows in step_latents.items()}
        for index in range(blocks):
            state.load_block(index)
            for step in hidden:
                hidden[step] = state.apply_block(hidden[step], step)
        return state.refined, {step: state.velocity(hidden[step], step) for step in hidden}


def f32_denoise(weights, inputs, noise_rows, scheduler, blocks=50):
    """The six-step Euler loop in float32, steps outer and blocks inner: six passes over the 50 mmap'ed blocks."""
    state = EditF32(weights, inputs)
    latents = noise_rows
    with torch.no_grad():
        for step in STEP_FIXTURE_STEPS:
            hidden = state.start(latents)
            for index in range(blocks):
                hidden = state.run_block(index, hidden, step)
            velocity = state.velocity(hidden, step)
            latents = scheduler.step(velocity, inputs["layout"]["plan"][step][0], latents, return_dict=False)[0]
            print(f"f32 denoise: step {step} done", flush=True)
    return latents


def stage_edit_f32truth(args, _device):
    """The float32 velocities of the edit_step inputs, one block at a time (the edit variant of the f32truth stage).

    Unlike the text-to-image stage it builds temb [2, 2688] from [t_step, 0.999] and passes the real adaln_indices
    (timestep_indices * 3 + token_tags) to every block call and the real timestep_indices to norm_out.
    """
    step_fixture = None if args.plan else load_fixture(args, "edit_step")
    inputs = edit_inputs(args, "step", step_fixture)
    layout = inputs["layout"]
    if args.plan:
        print_inputs_plan("edit_f32truth", inputs)
        show("temb (planned)", torch.empty(2, 2688))
        show("noise_pred_f32_stepN (planned)", torch.empty(layout["target_rows"], PATCH_ROWS))
        show("refined_text_f32 (planned)", torch.empty(layout["text_rows"], 5376))
        return
    assert torch.equal(step_fixture["adaln_indices"].long(), layout["adaln_indices"]), "adaln_indices differ from the fixture"
    assert torch.equal(step_fixture["timestep_indices"].long(), layout["plan"][0][2]), "timestep_indices differ from the fixture"
    weights = WeightMap(os.path.join(args.checkpoint, "transformer_diffusers"))
    refined, velocity = f32_edit_velocities(
        weights, inputs, {step: step_fixture[f"latents_in_step{step}"] for step in STEP_FIXTURE_STEPS}
    )
    tensors = {"refined_text_f32": refined}
    for step in STEP_FIXTURE_STEPS:
        tensors[f"noise_pred_f32_step{step}"] = velocity[step]
        tensors[f"step{step}"] = scalar(step)
        reference = step_fixture[f"noise_pred_step{step}"][layout["reference_rows"] :]
        error = (torch.linalg.norm(reference - velocity[step]) / torch.linalg.norm(velocity[step])).item()
        print(f"step {step}: bf16 reference vs f32 truth rel-L2 {error:.4g}")
    save_fixture(args, "edit_f32truth", {**tensors, **edit_settings(args, *inputs["canvas"], 1)})


def vision_after_smart_resize(image, processor):
    """The image the Qwen processor actually patchifies: the first-stage vision copy after smart_resize."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    ip = processor.image_processor
    width, height = image.size
    factor = ip.patch_size * ip.merge_size
    new_h, new_w = smart_resize(
        height, width, factor=factor, min_pixels=ip.size["shortest_edge"], max_pixels=ip.size["longest_edge"]
    )
    if (new_h, new_w) == (height, width):
        return image
    import torchvision.transforms.v2.functional as F

    resized = F.resize(
        pil_to_u8(image), [new_h, new_w], interpolation=F.InterpolationMode.BICUBIC, antialias=True
    )
    return Image.fromarray(resized.permute(1, 2, 0).numpy())


def stage_edit_e2e(args, device):
    """The whole edit on one 256x256 reference: VAE copy 1024x1024, 512x512 canvas, explicit noises, float32 truth.

    The text comes from the edit_text fixture (same image, prompt and seed), so the 62 GB encoder is not loaded twice.
    image_u8 is decoded from the float32 blockwise denoise; --bf16-reference adds image_u8_bf16 from the BF16
    transformer (hours on the CPU). The reference latents are those of edit_vae's reference 0 (same seed).
    """
    inputs = edit_inputs(args, "e2e")
    layout = inputs["layout"]
    if args.plan:
        print_inputs_plan("edit_e2e", inputs)
        show("vision_u8 after smart_resize", pil_to_u8(vision_after_smart_resize(inputs["text"]["vision_image"], inputs["text"]["processor"])))
        show("noise (planned)", torch.empty(LATENT_CHANNELS, *[side // VAE_RATIO for side in inputs["canvas"]]))
        show("image_u8 (planned)", torch.empty(3, *inputs["canvas"], dtype=torch.uint8))
        return
    noise_rows = edit_noise_rows(args, *inputs["canvas"])
    weights = WeightMap(os.path.join(args.checkpoint, "transformer_diffusers"))
    scheduler = make_scheduler(args, torch.device("cpu"))
    final_rows = f32_denoise(weights, inputs, noise_rows, scheduler)
    final = unpatchify(final_rows, layout["latent_height"], layout["latent_width"])[0]
    encoded = inputs["encoded"]
    tensors = {
        "image_u8": to_uint8(run_vae(args, final[None])[0])[0],
        "latents_final": final,
        "noise": noise_latents(args, *inputs["canvas"])[0, :, 0],
        "reference_posterior_noise_0": encoded["noise"],
        "reference_aug_noise_0": encoded["reference_noise"],
        "reference_latents_0": encoded["latents"],
        "condition_rows": inputs["condition_rows"],
        "reference_u8": pil_to_u8(inputs["text"]["reference"]),
        "vae_u8": pil_to_u8(inputs["text"]["vae_copy"]),
        "vision_first_u8": pil_to_u8(inputs["text"]["vision_image"]),
        "vision_u8": pil_to_u8(vision_after_smart_resize(inputs["text"]["vision_image"], inputs["text"]["processor"])),
        "token_ids": torch.tensor(inputs["text"]["token_ids"], dtype=torch.int32),
        "prompt": torch.tensor(list(args.prompt.encode("utf-8")), dtype=torch.uint8),
        **edit_settings(args, *inputs["canvas"], 1),
    }
    if args.bf16_reference:
        recorded = run_edit_transformer(args, device, inputs, noise_rows, last_step=5)
        final_bf16 = unpatchify(recorded["latents_final"], layout["latent_height"], layout["latent_width"])[0]
        tensors["image_u8_bf16"] = to_uint8(run_vae(args, final_bf16[None])[0])[0]
        tensors["latents_final_bf16"] = final_bf16
    save_fixture(args, "edit_e2e", tensors)


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
EDIT_PROMPT_STAGES = ("edit_text", "edit_text_small", "edit_step", "edit_f32truth", "edit_e2e")


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


if __name__ == "__main__":
    main()
