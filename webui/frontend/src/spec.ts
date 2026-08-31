import { CONSTANTS, QUALITY_PRESETS, SLOWER_FLAGS } from "./generated/options";
import type { JobSpec } from "./types";

export const DEFAULT_SPEC: JobSpec = {
  prompt: "",
  width: 512,
  height: 512,
  render_width: 0,
  render_height: 0,
  frames: null,
  seconds: 2.5,
  steps: 20,
  denoise_reuse: 1,
  dit_layers: 50,
  core_reuse: 1,
  token_reduction: false,
  ssd_streaming: false,
  use_int8_row_fc2: false,
  use_reference_rope: false,
  seed: 42,
  first_frame: null,
  last_frame: null,
  references: [],
  reference_image_size: "match",
  write_frames: false,
  profile: false,
  preview: true,
  slower: [],
  postprocess: [],
};

export const ALL_SLOWER_FLAGS = SLOWER_FLAGS;

/** Mirror of h3_align_frame_count: legal shapes are 5 + 17n. */
export function alignFrames(requested: number): number {
  const value = Math.max(requested, CONSTANTS.frames.align_base);
  const remainder = (value - CONSTANTS.frames.align_base) % CONSTANTS.frames.align_stride;
  return remainder === 0 ? value : value + CONSTANTS.frames.align_stride - remainder;
}

export function resolvedFrames(spec: JobSpec): number {
  const requested =
    spec.frames ??
    (spec.seconds !== null ? Math.round(spec.seconds * CONSTANTS.fps) : 56);
  return alignFrames(Math.max(requested, 1));
}

export function resolvedSeconds(spec: JobSpec): number {
  return resolvedFrames(spec) / CONSTANTS.fps;
}

export function megapixels(width: number, height: number): string {
  return (width * height / 1e6).toFixed(2);
}

export function applyQualityPreset(spec: JobSpec, id: string): JobSpec {
  const preset = QUALITY_PRESETS.find((entry) => entry.id === id);
  if (!preset) return spec;
  // A preset may also draw at a smaller size and enlarge: that is where most
  // of the time goes, so a "quick look" that only thins the model is not quick.
  const scale = preset.render_scale ?? 1;
  const grid = CONSTANTS.canvas_multiple;
  const snap = (value: number) => Math.max(grid, Math.round(value / grid) * grid);
  const render =
    scale < 1
      ? { render_width: snap(spec.width * scale), render_height: snap(spec.height * scale) }
      : { render_width: 0, render_height: 0 };
  return {
    ...spec,
    steps: preset.steps,
    dit_layers: preset.dit_layers,
    denoise_reuse: preset.denoise_reuse,
    core_reuse: 1,
    token_reduction: preset.token_reduction,
    ...render,
  };
}

export function matchingPreset(spec: JobSpec): string | null {
  const found = QUALITY_PRESETS.find((preset) => {
    const scaled = (preset.render_scale ?? 1) < 1;
    return (
      preset.steps === spec.steps &&
      preset.dit_layers === spec.dit_layers &&
      preset.denoise_reuse === spec.denoise_reuse &&
      preset.token_reduction === spec.token_reduction &&
      spec.core_reuse === 1 &&
      scaled === (spec.render_width > 0)
    );
  });
  return found?.id ?? null;
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "—";
  const total = Math.round(seconds);
  const minutes = Math.floor(total / 60);
  return `${String(minutes).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}
