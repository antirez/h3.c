from __future__ import annotations

from dataclasses import dataclass, replace

from .hardware import MacInfo


@dataclass(frozen=True, slots=True)
class GenerationPreset:
    name: str
    label: str
    description: str
    width: int
    height: int
    render_width: int | None
    render_height: int | None
    seconds: int
    steps: int
    layers: int
    reuse: int
    core_reuse: int
    ssd_streaming: bool
    live_preview: bool = False


PRESET_ORDER = ("fast", "balanced", "quality")

_PRESETS = {
    "fast": GenerationPreset(
            name="fast",
            label="Fast",
            description=(
                "Quickly check the prompt, identity, and framing with a smaller "
                "internal canvas."
            ),
            width=512,
            height=512,
            render_width=320,
            render_height=320,
            seconds=2,
            steps=6,
            layers=40,
            reuse=1,
            core_reuse=1,
            ssd_streaming=True,
    ),
    "balanced": GenerationPreset(
            name="balanced",
            label="Balanced",
            description="A good compromise for checking motion and likeness.",
            width=512,
            height=512,
            render_width=None,
            render_height=None,
            seconds=4,
            steps=20,
            layers=45,
            reuse=2,
            core_reuse=1,
            ssd_streaming=True,
    ),
    "quality": GenerationPreset(
            name="quality",
            label="Quality",
            description="Final render with every layer and full denoising.",
            width=512,
            height=512,
            render_width=None,
            render_height=None,
            seconds=6,
            steps=50,
            layers=50,
            reuse=1,
            core_reuse=1,
            ssd_streaming=True,
    ),
}


def recommended_preset_name(info: MacInfo) -> str:
    if info.memory_gib >= 64 and (
        "M5" in info.chip.upper() or "METAL 4" in info.metal_support.upper()
    ):
        return "balanced"
    return "fast"


def preset_for(name: str, info: MacInfo) -> GenerationPreset:
    try:
        preset = _PRESETS[name]
    except KeyError as error:
        raise ValueError(f"unknown preset: {name}") from error
    if name == "fast" and info.memory_gib < 32:
        return replace(
            preset,
            width=256,
            height=256,
            render_width=256,
            render_height=256,
            steps=4,
        )
    return preset
