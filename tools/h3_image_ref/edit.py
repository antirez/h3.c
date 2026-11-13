import copy
import os
import sys
import types

from PIL import Image, ImageOps
from diffusers import MiniMaxH3ImageTransformer2DModel, MiniMaxH3Scheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.modular_pipelines.minimax_h3_image.before_denoise import (
    build_image_packed_sequence,
    patchify_video_latents,
)
from diffusers.modular_pipelines.minimax_h3_image.before_encoder import bound_short_edge
from diffusers.modular_pipelines.minimax_h3_image.encoders import get_qwen3vl_prompt_embeds
from diffusers.utils.torch_utils import maybe_adjust_dtype_for_device, randn_tensor
import numpy as np
import torch
from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

from .common import (
    CANVAS_MULTIPLE,
    IMAGE_TAG,
    LATENT_CHANNELS,
    MAX_TOKENS,
    PATCH_SIZE,
    PIXEL_MEAN,
    PIXEL_STD,
    STEP_FIXTURE_STEPS,
    TEXT_LAYER,
    TEXT_TAG,
    VAE_RATIO,
    canvas,
    load_fixture,
    load_transformer_mmap,
    load_vae,
    make_scheduler,
    noise_latents,
    release,
    save_fixture,
    scalar,
    settings,
    show,
    to_uint8,
    unpatchify,
)
from .f32truth import WeightMap, f32_denoise, f32_edit_velocities
from .t2i import run_vae


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


EDIT_PROMPT_STAGES = ("edit_text", "edit_text_small", "edit_step", "edit_f32truth", "edit_e2e")
