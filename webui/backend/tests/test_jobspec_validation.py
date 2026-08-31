"""Table-driven check that the validator refuses exactly what h3 refuses."""

from pathlib import Path

import pytest

from app.argv import build_argv
from app.jobspec import JobSpec, Reference, align_frames, validate

BASE = {"prompt": "a fox", "width": 512, "height": 512, "frames": 22}


def errors_for(**overrides) -> list[str]:
    errors, _ = validate(JobSpec(**{**BASE, **overrides}))
    return errors


def test_a_plain_job_is_accepted():
    assert errors_for() == []


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"width": 500}, "width and height must be multiples of 32 and at least 32"),
        ({"height": 16}, "width and height must be multiples of 32 and at least 32"),
        (
            {"width": 1344, "height": 800},
            "canvas exceeds the released 768*1344 pixel limit",
        ),
        ({"render_width": 256}, "render width and height must be set together"),
        (
            {"render_width": 256, "render_height": 128},
            "internal render canvas must be same-aspect multiples of 32 "
            "no larger than the output canvas",
        ),
        (
            {"render_width": 1024, "render_height": 1024},
            "internal render canvas must be same-aspect multiples of 32 "
            "no larger than the output canvas",
        ),
        ({"frames": 4}, "frames must align within the released 5..362 range"),
        ({"frames": 363}, "frames must align within the released 5..362 range"),
        (
            {"frames": 5},
            "generation requires at least one trained 22-frame decoder chunk",
        ),
        ({"steps": 1}, "denoising steps must be in [2, 1000]"),
        ({"steps": 1001}, "denoising steps must be in [2, 1000]"),
        ({"denoise_reuse": 4}, "denoise reuse must be in [1, 3]"),
        ({"dit_layers": 34}, "DiT layers must be in [35, 50]"),
        ({"dit_layers": 51}, "DiT layers must be in [35, 50]"),
        ({"core_reuse": 7}, "core reuse must be in [1, 6]"),
        (
            {"core_reuse": 4, "denoise_reuse": 2},
            "core reuse and denoiser reuse cannot be combined",
        ),
        (
            {"frames": 22, "seconds": 1.0},
            "--seconds and --frames are mutually exclusive",
        ),
        (
            {"ssd_streaming": True, "use_int8_row_fc2": True},
            "SSD streaming uses original BF16 weights and cannot be combined "
            "with int8 row FC2",
        ),
        (
            {"use_int8_row_fc2": True, "slower": ["use-slower-bf16-mlp"]},
            "int8 row FC2 cannot be combined with the BF16 MLP",
        ),
        ({"prompt": "   "}, "a prompt is required"),
    ],
)
def test_rejected_jobs(overrides, message):
    assert message in errors_for(**overrides)


def test_int8_row_fc2_is_only_a_warning_on_cuda():
    errors, warnings = validate(JobSpec(**BASE, use_int8_row_fc2=True))
    assert errors == []
    assert any("no-op on this CUDA backend" in w for w in warnings)
    _, metal_warnings = validate(
        JobSpec(**BASE, use_int8_row_fc2=True), backend="metal"
    )
    assert metal_warnings == []


@pytest.mark.parametrize(
    ("requested", "aligned"),
    [(1, 5), (5, 5), (6, 22), (22, 22), (23, 39), (56, 56), (107, 107), (243, 243)],
)
def test_frame_alignment_matches_the_engine(requested, aligned):
    assert align_frames(requested) == aligned


def test_seconds_are_converted_and_rounded_up():
    assert JobSpec(**{**BASE, "frames": None, "seconds": 10.0}).resolved_frames() == 243
    assert JobSpec(**{**BASE, "frames": None, "seconds": 4.5}).resolved_frames() == 124
    # 4.4 s is 105.6 frames, which rounds up to the next legal shape, 107.
    assert JobSpec(**{**BASE, "frames": None, "seconds": 4.4}).resolved_frames() == 107


# ─────────────────────────────── references ────────────────────────────────

def image(name="fox.png") -> Reference:
    return Reference(kind="image", path=name)


def test_references_cannot_be_combined_with_anchors():
    assert "full references cannot be combined with frame anchors" in errors_for(
        references=[image()], first_frame="a.png"
    )


def test_at_most_twelve_references():
    assert "Ref2VA supports at most 12 references" in errors_for(
        references=[image(f"{n}.png") for n in range(13)]
    )


def test_per_kind_reference_limits():
    message = "Ref2VA limits are 9 images, 3 videos, and 3 audio inputs"
    assert message in errors_for(references=[image(f"{n}.png") for n in range(10)])
    assert message in errors_for(
        frames=56,
        references=[
            image(),
            *[Reference(kind="video", path=f"{n}.mp4") for n in range(4)],
        ],
    )


def test_audio_needs_a_visual_reference():
    assert "reference audio requires an image or video reference" in errors_for(
        references=[Reference(kind="audio", path="m.wav", seconds=6.0)]
    )


def test_audio_duration_rules():
    short = errors_for(
        references=[image(), Reference(kind="audio", path="m.wav", seconds=1.2)]
    )
    assert "reference audio requires at least 2 seconds at 32 kHz" in short
    long_total = errors_for(
        references=[
            image(),
            Reference(kind="audio", path="a.wav", seconds=8.0),
            Reference(kind="audio", path="b.wav", seconds=8.0),
        ]
    )
    assert "ordered reference audio exceeds 15 seconds in total" in long_total


def test_video_soundtrack_requires_at_least_56_frames():
    message = (
        "a video soundtrack requires at least 2 seconds; "
        "request at least 56 output frames"
    )
    assert message in errors_for(
        frames=22, references=[Reference(kind="video", path="clip.mp4")]
    )
    assert message not in errors_for(
        frames=56, references=[Reference(kind="video", path="clip.mp4")]
    )
    assert message not in errors_for(
        frames=22, references=[Reference(kind="silent_video", path="clip.mp4")]
    )


def test_video_audio_reference_needs_a_soundtrack():
    assert "video+audio reference 1 has no soundtrack path" in errors_for(
        frames=56, references=[Reference(kind="video_audio", path="clip.mp4")]
    )


# ────────────────────────────────── argv ───────────────────────────────────

def test_argv_is_a_list_and_never_a_shell_string():
    spec = JobSpec(**{**BASE, "prompt": 'a "quoted" fox; rm -rf /'})
    argv = build_argv(spec, Path("./h3"), Path("./MiniMax-H3"), Path("out.mp4"))
    assert argv[argv.index("-p") + 1] == 'a "quoted" fox; rm -rf /'
    assert all(isinstance(part, str) for part in argv)


def test_argv_resolves_duration_to_frames():
    spec = JobSpec(prompt="x", frames=None, seconds=4.5)
    argv = build_argv(spec, Path("h3"), Path("m"), Path("o.mp4"))
    assert "--seconds" not in argv
    assert argv[argv.index("--frames") + 1] == "124"


def test_argv_uses_core_reuse_instead_of_reuse_when_set():
    argv = build_argv(
        JobSpec(**BASE, core_reuse=4), Path("h3"), Path("m"), Path("o.mp4")
    )
    assert "--core-reuse" in argv and "--reuse" not in argv


def test_argv_keeps_reference_order_and_pairs_video_audio():
    spec = JobSpec(
        **{**BASE, "frames": 56},
        references=[
            image("first.png"),
            Reference(kind="video_audio", path="clip.mp4", audio_path="music.wav"),
            Reference(kind="audio", path="extra.wav", seconds=3.0),
        ],
    )
    argv = build_argv(spec, Path("h3"), Path("m"), Path("o.mp4"))
    tail = argv[argv.index("--ref-image-size") :]
    assert tail == [
        "--ref-image-size",
        "match",
        "--ref-image",
        "first.png",
        "--ref-video-audio",
        "clip.mp4",
        "music.wav",
        "--ref-audio",
        "extra.wav",
    ]


def test_argv_adds_preview_and_frames_directories_only_when_asked():
    plain = build_argv(JobSpec(**BASE), Path("h3"), Path("m"), Path("o.mp4"))
    assert "--preview-dir" not in plain and "--frames-dir" not in plain
    full = build_argv(
        JobSpec(**BASE),
        Path("h3"),
        Path("m"),
        Path("o.mp4"),
        frames_dir=Path("frames"),
        preview_dir=Path("preview"),
    )
    assert full[full.index("--preview-dir") + 1] == "preview"
    assert full[full.index("--frames-dir") + 1] == "frames"


def test_empty_output_disables_mp4_encoding():
    argv = build_argv(JobSpec(**BASE), Path("h3"), Path("m"), None)
    assert argv[argv.index("-o") + 1] == ""


def test_a_reference_clip_shorter_than_two_seconds_is_refused():
    message = "video soundtrack 1 requires at least 2 seconds: the clip is only 1.4 s"
    assert message in errors_for(
        frames=56, references=[Reference(kind="video", path="clip.mp4", seconds=1.4)]
    )
    assert message not in errors_for(
        frames=56,
        references=[Reference(kind="silent_video", path="clip.mp4", seconds=1.4)],
    )
