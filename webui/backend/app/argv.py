"""Build the exact argv for one `./h3` run.

An argv list, never a shell string: prompts and file names come from the
browser and must never be parsed by a shell. Durations are resolved to an
explicit frame count here, so the number shown in the UI is the number h3 runs.
"""

from pathlib import Path

from .jobspec import JobSpec, Reference

_REFERENCE_FLAG = {
    "image": "--ref-image",
    "video": "--ref-video",
    "silent_video": "--ref-silent-video",
    "video_audio": "--ref-video-audio",
    "audio": "--ref-audio",
}


def build_argv(
    spec: JobSpec,
    binary: Path,
    model_dir: Path,
    output: Path | None,
    frames_dir: Path | None = None,
    preview_dir: Path | None = None,
) -> list[str]:
    argv = [str(binary), "-d", str(model_dir), "-p", spec.prompt]
    argv += ["-o", str(output) if output else ""]
    argv += ["--width", str(spec.width), "--height", str(spec.height)]
    if spec.render_width and spec.render_height:
        argv += [
            "--render-width",
            str(spec.render_width),
            "--render-height",
            str(spec.render_height),
        ]
    argv += ["--frames", str(spec.resolved_frames())]
    argv += ["--steps", str(spec.steps)]
    argv += ["--layers", str(spec.dit_layers)]
    if spec.core_reuse > 1:
        argv += ["--core-reuse", str(spec.core_reuse)]
    else:
        argv += ["--reuse", str(spec.denoise_reuse)]
    if spec.token_reduction:
        argv.append("--token-reduction")
    if spec.ssd_streaming:
        argv.append("--ssd-streaming")
    if spec.use_int8_row_fc2:
        argv.append("--use-int8-row-fc2")
    if spec.use_reference_rope:
        argv.append("--use-reference-rope")
    for flag in spec.slower:
        argv.append(f"--{flag}")
    argv += ["--seed", str(spec.seed)]
    if spec.first_frame:
        argv += ["--first-frame", spec.first_frame]
    if spec.last_frame:
        argv += ["--last-frame", spec.last_frame]
    if any(reference.kind == "image" for reference in spec.references):
        argv += ["--ref-image-size", spec.reference_image_size]
    for reference in spec.references:
        argv += _reference_argv(reference)
    if frames_dir is not None:
        argv += ["--frames-dir", str(frames_dir)]
    if preview_dir is not None:
        argv += ["--preview-dir", str(preview_dir)]
    if spec.profile:
        argv.append("--profile")
    return argv


def _reference_argv(reference: Reference) -> list[str]:
    flag = _REFERENCE_FLAG[reference.kind]
    if reference.kind == "video_audio":
        return [flag, reference.path, reference.audio_path or ""]
    return [flag, reference.path]
