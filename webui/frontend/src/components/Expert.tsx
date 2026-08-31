import { CONSTANTS, OPTIONS } from "../generated/options";
import { ALL_SLOWER_FLAGS } from "../spec";
import type { JobSpec, Plugin, SystemInfo } from "../types";

interface Props {
  spec: JobSpec;
  system: SystemInfo | null;
  plugins: Plugin[];
  onChange: (spec: JobSpec) => void;
}

function Flag(props: {
  checked: boolean;
  onChange: (value: boolean) => void;
  title: string;
  flag: string;
  why?: string;
  disabled?: boolean;
}) {
  return (
    <div className={`check${props.disabled ? " off" : ""}`}>
      <input
        type="checkbox"
        checked={props.checked}
        disabled={props.disabled}
        onChange={(event) => props.onChange(event.target.checked)}
      />
      <span>
        {props.title} <code>{props.flag}</code>
        {props.why ? <span className="why">{props.why}</span> : null}
      </span>
    </div>
  );
}

/** Everything h3 accepts, named as it is on the command line. */
export function Expert({ spec, system, plugins, onChange }: Props) {
  const cuda = (system?.device?.architecture ?? "CUDA").startsWith("CUDA");
  const hidden = OPTIONS.filter((option) => option.ui === "hidden");

  return (
    <div className="inner">
      <div className="row">
        <div className="field">
          <span>
            Text and variation <span className="flag">--prompt --seed</span>
          </span>
          <p>
            The prompt is sent as written. The same prompt, settings and seed produce the
            same video on the same build.
          </p>
        </div>
        <div className="field">
          <span>
            Duration and canvas{" "}
            <span className="flag">
              --frames --seconds --width --height --render-width --render-height
            </span>
          </span>
          <p>
            Frames round up to 5 + 17n, {CONSTANTS.frames.min_generation}…
            {CONSTANTS.frames.max_aligned}. Sides are multiples of{" "}
            {CONSTANTS.canvas_multiple} and the area stays under{" "}
            {CONSTANTS.max_pixels_label}.
          </p>
        </div>
      </div>

      <div className="field">
        <span>
          Sampler{" "}
          <span className="flag">--steps --layers --reuse --core-reuse --token-reduction</span>
        </span>
        <p>
          Steps 2…{CONSTANTS.max_steps}, layers {CONSTANTS.dit_layers.min}…
          {CONSTANTS.dit_layers.max}, reuse 1…3, core reuse 1…6. Reuse and core reuse
          cannot both exceed 1.
        </p>
        <div className="row">
          <div>
            <span className="flag">--core-reuse</span>
            <input
              type="number"
              min={1}
              max={6}
              value={spec.core_reuse}
              onChange={(event) =>
                onChange({
                  ...spec,
                  core_reuse: Number(event.target.value) || 1,
                  denoise_reuse: Number(event.target.value) > 1 ? 1 : spec.denoise_reuse,
                })
              }
            />
          </div>
          <div>
            <span className="flag">--ref-image-size</span>
            <select
              value={spec.reference_image_size}
              onChange={(event) =>
                onChange({
                  ...spec,
                  reference_image_size: event.target.value as "match" | "max",
                })
              }
            >
              <option value="match">match</option>
              <option value="max">max</option>
            </select>
          </div>
        </div>
      </div>

      <div className="checks">
        <Flag
          checked={spec.ssd_streaming}
          title="SSD streaming"
          flag="--ssd-streaming"
          why="27.06 GB → 1.63 GB of GPU memory, about 38 % slower."
          onChange={(on) => onChange({ ...spec, ssd_streaming: on })}
        />
        <Flag
          checked={spec.use_int8_row_fc2}
          title="int8 row FC2"
          flag="--use-int8-row-fc2"
          disabled={cuda}
          why={
            cuda
              ? "Metal/M5 only — measured as a no-op on this CUDA backend."
              : "One activation scale per FC2 row."
          }
          onChange={(on) => onChange({ ...spec, use_int8_row_fc2: on })}
        />
        <Flag
          checked={spec.use_reference_rope}
          title="Reference RoPE"
          flag="--use-reference-rope"
          why="Restores the released 256 × 256 grid for parity checks."
          onChange={(on) => onChange({ ...spec, use_reference_rope: on })}
        />
        <Flag
          checked={spec.write_frames}
          title="Write frames"
          flag="--frames-dir"
          why="Every final frame as a PPM file."
          onChange={(on) => onChange({ ...spec, write_frames: on })}
        />
        <Flag
          checked={spec.profile}
          title="Profile phases"
          flag="--profile"
          why="Per-phase wall time and peak memory in the log."
          onChange={(on) => onChange({ ...spec, profile: on })}
        />
      </div>

      <div className="field" style={{ marginTop: 12 }}>
        <span>
          Parity flags <span className="flag">{ALL_SLOWER_FLAGS.length}</span>
        </span>
        <p>Force close-reference implementations, slower by design.</p>
        <div className="checks">
          {ALL_SLOWER_FLAGS.map((flag) => (
            <div className="check" key={flag}>
              <input
                type="checkbox"
                checked={spec.slower.includes(flag)}
                onChange={(event) =>
                  onChange({
                    ...spec,
                    slower: event.target.checked
                      ? [...spec.slower, flag]
                      : spec.slower.filter((entry) => entry !== flag),
                  })
                }
              />
              <span>
                <code>--{flag}</code>
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="field">
        <span>Post-processing</span>
        <div className="checks">
          {plugins.map((plugin) => (
            <Flag
              key={plugin.name}
              checked={spec.postprocess.includes(plugin.name)}
              disabled={!plugin.available}
              title={plugin.label}
              flag={plugin.name}
              why={`${plugin.description}${plugin.reason ? ` Unavailable: ${plugin.reason}.` : ""}`}
              onChange={(on) =>
                onChange({
                  ...spec,
                  postprocess: on
                    ? [...spec.postprocess, plugin.name]
                    : spec.postprocess.filter((name) => name !== plugin.name),
                })
              }
            />
          ))}
          {plugins.length === 0 ? (
            <p className="why">No post-processing plugin is registered.</p>
          ) : null}
        </div>
        {plugins.find((plugin) => plugin.notice) ? (
          <p className="why">{plugins.find((plugin) => plugin.notice)?.notice}</p>
        ) : null}
      </div>

      <div className="field">
        <span>
          Watching it being made <span className="flag">--preview-dir</span>
        </span>
        <p>
          Switched on under <b>Picture</b>. The server picks the directory the
          passes are written to.
        </p>
      </div>

      <div className="field">
        <span>
          The photos a video starts and ends on{" "}
          <span className="flag">--first-frame --last-frame</span>
        </span>
        <p>
          Chosen where they are seen, under <b>or start from · a photo</b>, next to
          the description. They are not repeated here.
        </p>
      </div>

      <div className="field">
        <span>
          Set by the server <span className="flag">--model-dir --output --info</span>
        </span>
        <p>
          Not exposed:{" "}
          {hidden
            .filter((option) => option.role === "excluded")
            .map((option) => option.flag)
            .join(" ")}{" "}
          — terminal graphics and CLI help, which a browser has no use for.
        </p>
      </div>
    </div>
  );
}
