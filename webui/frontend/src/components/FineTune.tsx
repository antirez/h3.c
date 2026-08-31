import { CONSTANTS } from "../generated/options";
import { humanMinutes } from "../copy";
import { resolvedFrames } from "../spec";
import type { JobSpec } from "../types";

interface Props {
  spec: JobSpec;
  deltas: Record<string, number | null>;
  onChange: (spec: JobSpec) => void;
}

function Delta({ seconds, base }: { seconds: number | null | undefined; base: number | null }) {
  if (seconds === null || seconds === undefined || base === null) return null;
  const difference = seconds - base;
  if (Math.abs(difference) < 20) return <span className="delta">about the same</span>;
  return (
    <span className="delta">
      {difference < 0 ? "saves" : "adds"} ≈ {humanMinutes(Math.abs(difference))}
    </span>
  );
}

/** The same trade-offs as Create, one control at a time and still in plain words. */
export function FineTune({ spec, deltas, onChange }: Props) {
  const base = deltas.base ?? null;
  const internal = spec.render_width > 0 && spec.render_height > 0;

  return (
    <div className="inner">
      <div className="row">
        <div className="field">
          <span>
            Detail passes
          </span>
          <p>How many times the picture is refined. More passes bring out more detail.</p>
          <input
            type="number"
            value={spec.steps}
            min={2}
            max={CONSTANTS.max_steps}
            onChange={(event) => onChange({ ...spec, steps: Number(event.target.value) || 2 })}
          />
          <Delta seconds={deltas.steps} base={base} />
        </div>
        <div className="field">
          <span>
            Model depth
          </span>
          <p>How much of the model runs. Less is faster and a little looser.</p>
          <input
            type="number"
            value={spec.dit_layers}
            min={CONSTANTS.dit_layers.min}
            max={CONSTANTS.dit_layers.max}
            onChange={(event) =>
              onChange({ ...spec, dit_layers: Number(event.target.value) || 50 })
            }
          />
          <Delta seconds={deltas.layers} base={base} />
        </div>
      </div>

      <div className="row">
        <div className="field">
          <span>
            How often it redraws
          </span>
          <p>
            Redrawing at every pass is closest to the reference. Less often is faster, and
            the framing can shift.
          </p>
          <select
            value={spec.core_reuse > 1 ? 0 : spec.denoise_reuse}
            onChange={(event) =>
              onChange({
                ...spec,
                denoise_reuse: Number(event.target.value) || 1,
                core_reuse: 1,
              })
            }
          >
            <option value={1}>Every pass — closest to the reference</option>
            <option value={2}>Every other pass — the validated fast setting</option>
            <option value={3}>Rarely — preview quality</option>
          </select>
          <Delta seconds={deltas.reuse} base={base} />
        </div>
        <div className="field">
          <span>
            Work smaller, then enlarge
          </span>
          <p>Draw at a smaller size and scale the result up. Faster, with less fine detail.</p>
          <select
            value={internal ? spec.render_width : 0}
            onChange={(event) => {
              const side = Number(event.target.value);
              const ratio = spec.height / spec.width;
              onChange({
                ...spec,
                render_width: side,
                render_height: side ? Math.round((side * ratio) / 32) * 32 : 0,
              });
            }}
          >
            <option value={0}>Off — draw at full size</option>
            <option value={384}>384 wide, then enlarge</option>
            <option value={320}>320 wide, then enlarge</option>
          </select>
          <Delta seconds={deltas.render} base={base} />
        </div>
      </div>

      <div className="row">
        <div className="field">
          <span>
            Exact size
          </span>
          <p>
            Multiples of 32, and at most {CONSTANTS.max_pixels_label} pixels in total.
          </p>
          <div className="row">
            <input
              type="number"
              step={32}
              value={spec.width}
              onChange={(event) => onChange({ ...spec, width: Number(event.target.value) || 0 })}
            />
            <input
              type="number"
              step={32}
              value={spec.height}
              onChange={(event) => onChange({ ...spec, height: Number(event.target.value) || 0 })}
            />
          </div>
        </div>
        <div className="field">
          <span>
            Exact length
          </span>
          <p>
            In frames rather than seconds. This job runs {resolvedFrames(spec)} frames.
          </p>
          <input
            type="number"
            value={spec.frames ?? resolvedFrames(spec)}
            onChange={(event) =>
              onChange({ ...spec, frames: Number(event.target.value) || null, seconds: null })
            }
          />
        </div>
      </div>

      <div className="check">
        <input
          type="checkbox"
          checked={spec.preview}
          onChange={(event) => onChange({ ...spec, preview: event.target.checked })}
        />
        <span>
          Watch it being made
          <span className="why">
            Shows the picture after every pass, so you can stop early if it is going the
            wrong way.
          </span>
        </span>
      </div>
      <div className="check">
        <input
          type="checkbox"
          checked={spec.token_reduction}
          onChange={(event) => onChange({ ...spec, token_reduction: event.target.checked })}
        />
        <span>
          Pair up detail while drawing
          <span className="why">
            Faster, and the composition can drift. Leave it off at small sizes.
          </span>
        </span>
      </div>
    </div>
  );
}
