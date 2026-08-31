"""Job specification and the validation h3 would otherwise refuse at runtime.

Every message here is copied verbatim from h3.c or main.c, so what the browser
shows before submitting is what the engine would have said afterwards.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

FPS = 24
CANVAS_MULTIPLE = 32
MAX_PIXELS = 768 * 1344
MIN_FRAMES_GENERATION = 22
MAX_FRAMES = 362
# A truncated video soundtrack needs 2 s, and 39 frames is only 1.625 s.
MIN_FRAMES_FOR_SOUNDTRACK = 56
MAX_REFERENCE_AUDIO_SECONDS = 15.0
MIN_REFERENCE_AUDIO_SECONDS = 2.0

ReferenceKind = Literal["image", "video", "silent_video", "video_audio", "audio"]


def align_frames(requested: int) -> int:
    """Mirror of h3_align_frame_count: legal shapes are 5 + 17n."""
    value = max(requested, 5)
    remainder = (value - 5) % 17
    return value if remainder == 0 else value + 17 - remainder


def frames_from_seconds(seconds: float) -> int:
    return round(seconds * FPS)


class Reference(BaseModel):
    kind: ReferenceKind
    path: str
    # Only for video_audio: the replacement soundtrack.
    audio_path: str | None = None
    # Filled from the asset store; used for the audio duration rules.
    seconds: float | None = None


class JobSpec(BaseModel):
    prompt: str = ""
    width: int = 864
    height: int = 480
    render_width: int = 0
    render_height: int = 0
    frames: int | None = None
    seconds: float | None = None
    steps: int = 20
    denoise_reuse: int = 1
    dit_layers: int = 50
    core_reuse: int = 1
    token_reduction: bool = False
    ssd_streaming: bool = False
    use_int8_row_fc2: bool = False
    use_reference_rope: bool = False
    seed: int = 42
    first_frame: str | None = None
    last_frame: str | None = None
    references: list[Reference] = Field(default_factory=list)
    reference_image_size: Literal["match", "max"] = "match"
    write_frames: bool = False
    profile: bool = False
    preview: bool = False
    slower: list[str] = Field(default_factory=list)
    postprocess: list[str] = Field(default_factory=list)

    def resolved_frames(self) -> int:
        """The frame count h3 will actually generate."""
        requested = (
            self.frames
            if self.frames is not None
            else frames_from_seconds(self.seconds)
            if self.seconds is not None
            else 56
        )
        return align_frames(max(requested, 1))


class EstimateRequest(BaseModel):
    """A job, plus the alternatives the interface wants labelled with a time."""

    spec: JobSpec
    variants: list[dict[str, Any]] = Field(default_factory=list)


def validate(spec: JobSpec, backend: str = "cuda") -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors mean h3 would refuse the job."""
    errors: list[str] = []
    warnings: list[str] = []

    if not spec.prompt.strip():
        errors.append("a prompt is required")

    _check_canvas(spec, errors)
    _check_duration(spec, errors)
    _check_sampler(spec, errors)
    _check_backend_flags(spec, backend, errors, warnings)
    _check_references(spec, errors)
    return errors, warnings


def _check_canvas(spec: JobSpec, errors: list[str]) -> None:
    if (
        spec.width < CANVAS_MULTIPLE
        or spec.height < CANVAS_MULTIPLE
        or spec.width % CANVAS_MULTIPLE
        or spec.height % CANVAS_MULTIPLE
    ):
        errors.append("width and height must be multiples of 32 and at least 32")
    if spec.width * spec.height > MAX_PIXELS:
        errors.append("canvas exceeds the released 768*1344 pixel limit")
    if (spec.render_width == 0) != (spec.render_height == 0):
        errors.append("render width and height must be set together")
    elif spec.render_width and (
        spec.render_width < CANVAS_MULTIPLE
        or spec.render_height < CANVAS_MULTIPLE
        or spec.render_width % CANVAS_MULTIPLE
        or spec.render_height % CANVAS_MULTIPLE
        or spec.render_width > spec.width
        or spec.render_height > spec.height
        or spec.render_width * spec.height != spec.render_height * spec.width
    ):
        errors.append(
            "internal render canvas must be same-aspect multiples of 32 "
            "no larger than the output canvas"
        )


def _check_duration(spec: JobSpec, errors: list[str]) -> None:
    if spec.frames is not None and spec.seconds is not None:
        errors.append("--seconds and --frames are mutually exclusive")
    if spec.seconds is not None and spec.seconds <= 0:
        errors.append("invalid seconds")
        return
    requested = spec.frames if spec.frames is not None else None
    if requested is not None and requested < 5:
        errors.append("frames must align within the released 5..362 range")
        return
    aligned = spec.resolved_frames()
    if aligned > MAX_FRAMES:
        errors.append("frames must align within the released 5..362 range")
    elif aligned < MIN_FRAMES_GENERATION:
        errors.append("generation requires at least one trained 22-frame decoder chunk")


def _check_sampler(spec: JobSpec, errors: list[str]) -> None:
    if not 2 <= spec.steps <= 1000:
        errors.append("denoising steps must be in [2, 1000]")
    if not 1 <= spec.denoise_reuse <= 3:
        errors.append("denoise reuse must be in [1, 3]")
    if not 35 <= spec.dit_layers <= 50:
        errors.append("DiT layers must be in [35, 50]")
    if not 1 <= spec.core_reuse <= 6:
        errors.append("core reuse must be in [1, 6]")
    if spec.core_reuse > 1 and spec.denoise_reuse > 1:
        errors.append("core reuse and denoiser reuse cannot be combined")


def _check_backend_flags(
    spec: JobSpec, backend: str, errors: list[str], warnings: list[str]
) -> None:
    if spec.ssd_streaming and spec.use_int8_row_fc2:
        errors.append(
            "SSD streaming uses original BF16 weights and cannot be combined "
            "with int8 row FC2"
        )
    if spec.use_int8_row_fc2 and "use-slower-bf16-mlp" in spec.slower:
        errors.append("int8 row FC2 cannot be combined with the BF16 MLP")
    if spec.use_int8_row_fc2 and backend == "cuda":
        warnings.append(
            "--use-int8-row-fc2 is a Metal/M5 specialization and a measured "
            "no-op on this CUDA backend"
        )


def _check_references(spec: JobSpec, errors: list[str]) -> None:
    references = spec.references
    if not references:
        return
    if spec.first_frame or spec.last_frame:
        errors.append("full references cannot be combined with frame anchors")
    if len(references) > 12:
        errors.append("Ref2VA supports at most 12 references")

    video_kinds = ("video", "silent_video", "video_audio")
    images = sum(1 for r in references if r.kind == "image")
    videos = sum(1 for r in references if r.kind in video_kinds)
    # A plain --ref-video keeps its embedded audio, so it counts as an input;
    # --ref-silent-video does not.
    audio_inputs = sum(
        1 for r in references if r.kind in ("audio", "video", "video_audio")
    )
    if images > 9 or videos > 3 or audio_inputs > 3:
        errors.append("Ref2VA limits are 9 images, 3 videos, and 3 audio inputs")
    if not any(r.kind != "audio" for r in references):
        errors.append("reference audio requires an image or video reference")

    for index, reference in enumerate(references, start=1):
        if reference.kind == "video_audio" and not reference.audio_path:
            errors.append(f"video+audio reference {index} has no soundtrack path")

    # A video soundtrack is truncated to min(clip length, output length), and
    # h3 refuses anything shorter than two seconds.
    output_seconds = spec.resolved_frames() / FPS
    has_soundtrack = any(r.kind in ("video", "video_audio") for r in references)
    if has_soundtrack and output_seconds < MIN_REFERENCE_AUDIO_SECONDS:
        errors.append(
            "a video soundtrack requires at least 2 seconds; "
            "request at least 56 output frames"
        )
    for index, reference in enumerate(references, start=1):
        if reference.kind not in ("video", "video_audio"):
            continue
        if (
            reference.seconds is not None
            and reference.seconds < MIN_REFERENCE_AUDIO_SECONDS
        ):
            errors.append(
                f"video soundtrack {index} requires at least 2 seconds: "
                f"the clip is only {reference.seconds:g} s"
            )

    total = 0.0
    for reference in references:
        if reference.kind != "audio":
            continue
        if reference.seconds is None:
            continue
        if reference.seconds < MIN_REFERENCE_AUDIO_SECONDS:
            errors.append("reference audio requires at least 2 seconds at 32 kHz")
        total += reference.seconds
    if total > MAX_REFERENCE_AUDIO_SECONDS:
        errors.append("ordered reference audio exceeds 15 seconds in total")
