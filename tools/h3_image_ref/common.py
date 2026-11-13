import gc
import json
import mmap
import os
import struct
import sys

from diffusers import AutoencoderKLMiniMaxH3Image, MiniMaxH3ImageTransformer2DModel, MiniMaxH3Scheduler
from diffusers.models.transformers.transformer_minimax_h3_image import MiniMaxH3ImageRotaryPosEmbed
from diffusers.modular_pipelines.minimax_h3_image.before_denoise import (
    build_image_packed_sequence,
    get_image_canvas_size,
)
from diffusers.utils.torch_utils import maybe_adjust_dtype_for_device, randn_tensor
from safetensors.torch import load_file, save_file
import torch


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


def unpatchify(rows, latent_height, latent_width):
    patch_h, patch_w = PATCH_SIZE[1], PATCH_SIZE[2]
    rows = rows.reshape(-1, latent_height // patch_h, latent_width // patch_w, LATENT_CHANNELS, patch_h, patch_w)
    rows = rows.permute(0, 3, 1, 4, 2, 5)
    return rows.reshape(-1, LATENT_CHANNELS, latent_height, latent_width).contiguous()


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
