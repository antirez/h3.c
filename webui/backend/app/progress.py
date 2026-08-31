"""Weighted progress and ETA.

A naive bar would jump: on the calibration run `load transformer core` took
40.9 s and `denoise` 5.1 s, so counting phases equally is misleading. Weights
come from a real measured run (webui/shared/progress_weights.json, produced by
tools/calibrate_progress.py) and are scaled to the job's own settings. The ETA
then corrects itself with what the current run has actually taken so far.
"""

import json
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any

from .db import Database
from .jobspec import JobSpec

# What each phase's cost is proportional to. A phase not listed here is a fixed
# cost: loading the text encoder takes the same time whatever the video is.
SCALES: dict[str, tuple[str, ...]] = {
    "denoise": ("steps", "pixels", "frames"),
    "denoise enqueue": ("steps", "pixels", "frames"),
    # The decode and the mux happen inside this phase's wall time.
    "video VAE load": ("pixels", "frames"),
    "FFmpeg": ("frames",),
}
UNKNOWN_PHASE_SECONDS = 1.0


@lru_cache
def load_weights(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _work(phase: str, spec_like: dict[str, Any]) -> float:
    """Absolute amount of work a phase does for one configuration.

    Pixels use the internal canvas when there is one, because that is what the
    model and the VAE actually run at.
    """
    width = spec_like.get("render_width") or spec_like["width"]
    height = spec_like.get("render_height") or spec_like["height"]
    amount = {
        "steps": spec_like["steps"],
        "pixels": width * height / 1e6,
        "frames": spec_like["frames"],
    }
    work = 1.0
    for dimension in SCALES.get(phase, ()):
        work *= amount[dimension]
    return work


class ProgressModel:
    def __init__(self, weights: dict[str, Any]) -> None:
        self.reference = weights["reference"]
        self.phase_seconds: dict[str, float] = weights["phase_seconds"]
        self.order = list(self.phase_seconds)
        self.factors: dict[str, Any] = weights.get("factors", {})
        self.fit = _fit(weights.get("samples", []))

    def plan(self, spec: JobSpec) -> list[tuple[str, float]]:
        """Expected seconds per phase for this job, in order."""
        ratios = self._ratios(spec)
        shape = {
            "width": spec.width,
            "height": spec.height,
            "render_width": spec.render_width,
            "render_height": spec.render_height,
            "steps": spec.steps,
            "frames": spec.resolved_frames(),
        }
        plan = []
        for phase in self.order:
            if phase in self.fit:
                fixed, variable = self.fit[phase]
                seconds = fixed + variable * _work(phase, shape)
            else:
                seconds = self.phase_seconds[phase]
                for dimension in SCALES.get(phase, ()):
                    seconds *= ratios[dimension]
            if phase.startswith("denoise"):
                seconds *= self._denoise_factor(spec)
                seconds += self._streaming_seconds(spec)
            elif phase == "load transformer core" and spec.ssd_streaming:
                seconds *= self._factor("ssd_streaming", "load_factor", 1.0)
            plan.append((phase, seconds))
        if spec.preview and "preview VAE load" not in self.phase_seconds:
            # Enabling the preview adds a VAE load; charge it like the decoder.
            index = next(
                (i for i, (name, _) in enumerate(plan) if name == "denoise"), len(plan)
            )
            plan.insert(index, ("preview VAE load", self._preview_seconds(spec)))
        return plan

    def fraction(self, spec: JobSpec, phase: str | None, completed: int, total: int
                 ) -> float:
        plan = self.plan(spec)
        budget = sum(seconds for _, seconds in plan)
        if budget <= 0:
            return 0.0
        done = 0.0
        for name, seconds in plan:
            if name == phase:
                share = completed / total if total else 0.0
                return min(1.0, (done + seconds * min(max(share, 0.0), 1.0)) / budget)
            done += seconds
        # An unknown phase carries no information about position: report
        # nothing and let the caller keep the highest value seen so far.
        return 0.0

    def remaining_seconds(
        self, spec: JobSpec, phase: str | None, completed: int, total: int,
        elapsed: float
    ) -> float | None:
        """Estimate what is left, corrected by how this run is actually going."""
        share = self.fraction(spec, phase, completed, total)
        if share <= 0.02 or elapsed <= 0:
            budget = sum(seconds for _, seconds in self.plan(spec))
            return max(budget - elapsed, 0.0) if budget else None
        return max(elapsed / share - elapsed, 0.0)

    def _denoise_factor(self, spec: JobSpec) -> float:
        """What the sampler settings do to the cost of one pass.

        Steps, pixels and frames are already in the ratios; this is everything
        else the quality presets change, which is most of what they change.
        """
        reference_layers = self._factor("dit_layers", "reference", 50) or 50
        factor = spec.dit_layers / reference_layers
        if spec.core_reuse > 1:
            heads = self._factor("core_reuse", "head_share", 0.3)
            factor *= heads + (1 - heads) / spec.core_reuse
        else:
            reuse = self.factors.get("denoise_reuse", {})
            factor *= float(reuse.get(str(spec.denoise_reuse), 1.0))
        if spec.token_reduction:
            factor *= self._factor("token_reduction", "factor", 1.0)
        return factor

    def _streaming_seconds(self, spec: JobSpec) -> float:
        """Streaming the weights from disk costs the same on every step."""
        if not spec.ssd_streaming:
            return 0.0
        return self._factor("ssd_streaming", "added_seconds_per_step", 0.0) * spec.steps

    def _factor(self, group: str, key: str, fallback: float) -> float:
        return float(self.factors.get(group, {}).get(key, fallback))

    def _preview_seconds(self, spec: JobSpec) -> float:
        return self.phase_seconds.get("video VAE load", 10.0) * 0.5 * self._ratios(
            spec
        )["pixels"]

    def _ratios(self, spec: JobSpec) -> dict[str, float]:
        reference = self.reference
        width = spec.render_width or spec.width
        height = spec.render_height or spec.height
        return {
            "steps": spec.steps / max(reference["steps"], 1),
            "pixels": (width * height)
            / max(reference["width"] * reference["height"], 1),
            "frames": spec.resolved_frames() / max(reference["frames"], 1),
        }


def _fit(samples: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Split each phase into a fixed and a per-unit-of-work cost.

    One calibration run cannot tell the two apart: with a single sample every
    phase looks purely proportional, and the quality presets then all cost the
    same. Two runs at different sizes separate them.
    """
    if len(samples) < 2:
        return {}
    first, last = samples[0], samples[-1]
    fitted: dict[str, tuple[float, float]] = {}
    for phase in first["phase_seconds"]:
        if phase not in last["phase_seconds"]:
            continue
        w1 = _work(phase, first["reference"])
        w2 = _work(phase, last["reference"])
        s1 = first["phase_seconds"][phase]
        s2 = last["phase_seconds"][phase]
        if abs(w2 - w1) < 1e-9:
            fitted[phase] = (min(s1, s2), 0.0)
            continue
        variable = (s2 - s1) / (w2 - w1)
        fixed = s1 - variable * w1
        if variable < 0:
            # Noise, not a real saving: treat the phase as a fixed cost.
            fitted[phase] = (min(s1, s2), 0.0)
        elif fixed < 0:
            # All of it scales; anchor on the larger, more reliable sample.
            fitted[phase] = (0.0, s2 / w2)
        else:
            fitted[phase] = (fixed, variable)
    return fitted


# How far a learned correction may pull the estimate. Beyond this the history
# is telling us something the model cannot express, and a wrong number with
# confidence is worse than a rough one.
CORRECTION_RANGE = (0.25, 4.0)
CORRECTION_SAMPLE = 12


def observed_correction(
    database: Database, model: "ProgressModel"
) -> tuple[float, int]:
    """How wrong the estimate has been lately, as a single factor.

    Two calibration runs fix the shape of the model, not its accuracy across
    every size: the cost of drawing does not grow linearly forever. Rather than
    pretend otherwise, the estimate is scaled by what recent jobs on this
    machine actually took.
    """
    rows = database.query_all(
        "SELECT params, started_at, finished_at FROM jobs "
        "WHERE state = 'completed' AND started_at IS NOT NULL "
        "AND finished_at IS NOT NULL ORDER BY id DESC LIMIT ?",
        (CORRECTION_SAMPLE,),
    )
    ratios: list[float] = []
    for row in rows:
        try:
            spec = JobSpec.model_validate(json.loads(row["params"]))
        except (ValueError, TypeError):
            continue
        predicted = sum(seconds for _, seconds in model.plan(spec))
        actual = _seconds_between(row["started_at"], row["finished_at"])
        if predicted > 1 and actual > 1:
            ratios.append(actual / predicted)
    if len(ratios) < 2:
        return 1.0, len(ratios)
    low, high = CORRECTION_RANGE
    return min(max(median(ratios), low), high), len(ratios)


def _seconds_between(started: str, finished: str) -> float:
    from datetime import UTC, datetime

    fmt = "%Y-%m-%d %H:%M:%S"
    begin = datetime.strptime(started, fmt).replace(tzinfo=UTC)
    end = datetime.strptime(finished, fmt).replace(tzinfo=UTC)
    return (end - begin).total_seconds()
