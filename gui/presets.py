from __future__ import annotations

from dataclasses import dataclass

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


def recommended_preset_name(info: MacInfo) -> str:
    del info
    return "fast"


def preset_for(name: str, info: MacInfo) -> GenerationPreset:
    del info
    presets = {
        "fast": GenerationPreset(
            name="fast",
            label="Veloce",
            description=(
                "Controlla rapidamente prompt, identità e inquadratura con "
                "un canvas interno ridotto."
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
            label="Bilanciato",
            description="Buon compromesso per verificare movimento e somiglianza.",
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
            label="Qualità",
            description="Render finale con tutti i layer e denoising completo.",
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
    try:
        return presets[name]
    except KeyError as error:
        raise ValueError(f"preset sconosciuto: {name}") from error
