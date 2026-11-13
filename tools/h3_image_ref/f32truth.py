import json
import os

from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.transformers.transformer_minimax_h3_image import (
    MiniMaxH3ImageAdaLayerNormOut,
    MiniMaxH3ImageRotaryPosEmbed,
    MiniMaxH3ImageTokenRefiner,
    MiniMaxH3ImageTransformerBlock,
)
from diffusers.modular_pipelines.minimax_h3_image.before_denoise import patchify_video_latents
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from .common import (
    BLOCK_ARGS,
    PATCH_SIZE,
    SAMPLE_SIGMAS,
    STEP_FIXTURE_STEPS,
    VAE_RATIO,
    canvas,
    load_fixture,
    noise_latents,
    packed_layout,
    save_fixture,
    scalar,
    settings,
)


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
