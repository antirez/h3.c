import { api } from "../api";
import { clock, explain, humanMinutes, phaseName, railIndex, railPhases } from "../copy";
import { resolvedSeconds } from "../spec";
import type { Job } from "../types";

/** The signature moment: the picture emerging from noise, pass by pass.
 *
 *  The same stage shows a finished take, with the video where the developing
 *  frame was: what you watched being made is what you watch afterwards.
 */
export function RenderStage(props: {
  job: Job;
  onStop: () => void;
  onLeave?: () => void;
}) {
  const { job, onStop, onLeave } = props;
  const running = job.state === "running" || job.state === "queued";
  const phases = railPhases(job);
  const reached = running ? railIndex(phases, job.phase) : phases.length;
  const preview =
    job.params.preview && job.preview_step !== null
      ? api.previewUrl(job.id, job.preview_step)
      : null;
  const broken = job.state === "failed" || job.state === "cancelled";

  return (
    <>
      <div className="summary">
        <span className="quote">“{job.prompt}”</span>
        <span className="meta">
          {job.params.width}×{job.params.height} · variation {job.params.seed}
        </span>
        {running && onLeave ? (
          <button className="stop" onClick={onLeave}>
            Keep making
          </button>
        ) : null}
        <button className="stop" onClick={onStop}>
          {running ? "Stop" : "Close"}
        </button>
      </div>

      <div className="develop">
        <div
          className="frame"
          style={{ aspectRatio: `${job.params.width} / ${job.params.height}` }}
        >
          {job.state === "completed" ? (
            <video src={api.videoUrl(job.id)} controls autoPlay loop muted />
          ) : broken ? (
            <div className="spinner" style={{ display: "grid", placeItems: "center", height: "100%", padding: 24, textAlign: "center" }}>
              {job.error ? explain(job.error).title : "This one did not finish."}
            </div>
          ) : preview ? (
            <img src={preview} alt={`pass ${(job.preview_step ?? 0) + 1}`} />
          ) : (
            <div className="spinner" style={{ display: "grid", placeItems: "center", height: "100%" }}>
              {job.params.preview
                ? "the first pass has not been drawn yet"
                : "watching is turned off for this video"}
            </div>
          )}
          {running && preview ? (
            <span className="stepmark">
              pass {(job.preview_step ?? 0) + 1} of {job.params.steps}
            </span>
          ) : null}
        </div>

        {/* The film edge doubles as the progress bar: one line, not three. */}
        <div className="rail" aria-hidden="true">
          {phases.map((phase, index) => (
            <span
              key={phase}
              title={phaseName(phase)}
              className={`perf${
                reached < 0 ? "" : index < reached ? " done" : index === reached ? " now" : ""
              }`}
            />
          ))}
        </div>
      </div>

      <div className="phase">
        {running ? (
          <>
            <div className="said">
              {phaseName(job.phase)}
              {job.phase?.startsWith("denoise") && job.total
                ? ` — pass ${Math.min(job.completed + 1, job.total)} of ${job.total}`
                : ""}
            </div>
            <div className="clock">
              <span>
                elapsed <b>{clock(job.elapsed)}</b>
              </span>
              {job.remaining !== null ? (
                <span>
                  about <b>{humanMinutes(job.remaining)}</b> left
                </span>
              ) : null}
              <span>{Math.round(job.progress * 100)} % done</span>
            </div>
          </>
        ) : broken ? (
          <>
            <div className="said">
              {job.error ? explain(job.error).title : "This one did not finish."}
            </div>
            <div className="clock">
              <span>{job.error ? explain(job.error).fix : ""}</span>
            </div>
          </>
        ) : (
          <>
            <div className="said">Ready</div>
            <div className="clock">
              <span>
                {resolvedSeconds(job.params).toFixed(1)} s · {job.params.width}×
                {job.params.height}
              </span>
              <span>
                made in <b>{clock(job.elapsed)}</b>
              </span>
              <span>variation {job.params.seed}</span>
            </div>
          </>
        )}
        <div className="tech">
          {job.phase ?? "starting"} {job.total ? `${job.completed}/${job.total}` : ""} ·{" "}
          {job.params.width}×{job.params.height} · seed {job.params.seed}
        </div>
      </div>
    </>
  );
}
