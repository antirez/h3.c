import copy
import os
import sys

from diffusers import MiniMaxH3ImageTransformer2DModel
from diffusers.modular_pipelines.minimax_h3_image.before_denoise import patchify_video_latents
from diffusers.utils.torch_utils import randn_tensor
from safetensors.torch import load_file
import torch
from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from .common import (
    FIXTURE_NAMES,
    LATENT_CHANNELS,
    MAX_TOKENS,
    PATCH_SIZE,
    STEP_FIXTURE_STEPS,
    StopForward,
    TEXT_LAYER,
    VAE_RATIO,
    canvas,
    decode_latents,
    load_fixture,
    load_transformer_mmap,
    load_vae,
    make_scheduler,
    noise_latents,
    packed_layout,
    release,
    save_fixture,
    scalar,
    settings,
    to_uint8,
    unpatchify,
)


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
