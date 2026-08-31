import { api } from "../api";
import { clock, humanMinutes, phaseName } from "../copy";
import type { Job } from "../types";

/** A video keeps being made while you set up the next one.
 *
 *  This is what stays on screen once you leave the stage: enough to know how
 *  it is going, and one click to go back and watch.
 */
export function LiveStrip({ job, onWatch, onStop }: {
  job: Job;
  onWatch: () => void;
  onStop: () => void;
}) {
  const preview =
    job.params.preview && job.preview_step !== null
      ? api.previewUrl(job.id, job.preview_step)
      : null;

  return (
    <div className="live">
      <button className="peek" onClick={onWatch} aria-label="Watch it being made">
        {preview ? <img src={preview} alt="" /> : <span className="spinner">…</span>}
      </button>
      <div className="what">
        <div className="t">{job.prompt || "(no description)"}</div>
        <div className="meter">
          <span style={{ width: `${Math.round(job.progress * 100)}%` }} />
        </div>
        <div className="clock">
          <span>{phaseName(job.phase)}</span>
          <span>
            {clock(job.elapsed)}
            {job.remaining !== null ? ` · about ${humanMinutes(job.remaining)} left` : ""}
          </span>
        </div>
      </div>
      <div className="acts">
        <button className="lib" onClick={onWatch}>
          Watch
        </button>
        <button className="lib" onClick={onStop}>
          Stop
        </button>
      </div>
    </div>
  );
}
