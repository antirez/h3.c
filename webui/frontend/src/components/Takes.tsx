import { useRef, useState } from "react";

import { api } from "../api";
import { clock, stateName } from "../copy";
import type { Job } from "../types";
import { DeleteControl } from "./DeleteControl";

interface Props {
  jobs: Job[];
  onOpen: (job: Job) => void;
  onDelete: (job: Job) => void;
}

/** Everything this machine has made, most recent first.
 *
 *  R29 P3: a grid, and hovering a take starts it playing in place; the
 *  actions surface over the moving picture instead of waiting below it.
 */
export function Takes({ jobs, onOpen, onDelete }: Props) {
  const done = jobs.filter((job) => job.state === "completed");
  return (
    <section className="takes">
      <h2>Takes</h2>
      {done.length === 0 ? (
        <div className="empty">
          Nothing made yet. Describe a scene above and the first take lands here.
        </div>
      ) : (
        <div className="grid">
          {done.map((job) => (
            <TakeCard key={job.id} job={job} onOpen={onOpen} onDelete={onDelete} />
          ))}
        </div>
      )}
    </section>
  );
}

function TakeCard({ job, onOpen, onDelete }: {
  job: Job;
  onOpen: (job: Job) => void;
  onDelete: (job: Job) => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  // A take whose video cannot be read has no poster either (T107): the
  // cell says so, instead of showing a broken picture.
  const [broken, setBroken] = useState(false);

  return (
    <div
      className="take"
      onMouseEnter={() => {
        const player = video.current;
        if (player) {
          player.currentTime = 0;
          // A poster stays up if the video cannot play yet.
          void player.play().catch(() => {});
        }
      }}
      onMouseLeave={() => video.current?.pause()}
    >
      <div className="screen" onClick={() => onOpen(job)}>
        {broken ? (
          <div className="poster gone">no picture</div>
        ) : (
          <img
            className="poster"
            src={api.posterUrl(job.id)}
            alt=""
            onError={() => setBroken(true)}
          />
        )}
        <video
          ref={video}
          className="playing"
          src={api.videoUrl(job.id)}
          muted
          loop
          playsInline
          preload="metadata"
        />
        <div className="veil" onClick={(event) => event.stopPropagation()}>
          <button onClick={() => onOpen(job)}>Open</button>
          <a href={api.videoUrl(job.id)} download>
            Save
          </a>
          <DeleteControl label="Delete" onDelete={() => onDelete(job)} />
        </div>
      </div>
      <span className="cap">{job.prompt.split(" ").slice(0, 5).join(" ")}</span>
      <span className="meta">
        {job.params.width}×{job.params.height} · {clock(job.elapsed)}
      </span>
    </div>
  );
}

/** Jobs that are not the one on stage: waiting, stopped or broken. */
export function Waiting({ jobs, onCancel, onLog }: {
  jobs: Job[];
  onCancel: (id: number) => void;
  onLog: (job: Job) => void;
}) {
  const others = jobs.filter(
    (job) => job.state === "queued" || job.state === "failed" || job.state === "cancelled",
  );
  if (others.length === 0) return null;
  return (
    <div className="queued">
      {others.slice(0, 6).map((job) => (
        <div className="one" key={job.id}>
          <span className="st">{stateName(job.state)}</span>
          <span className="t">{job.prompt || "(no description)"}</span>
          {job.state === "queued" ? (
            <button className="lib" onClick={() => onCancel(job.id)}>
              remove
            </button>
          ) : (
            <button className="lib" onClick={() => onLog(job)}>
              what happened
            </button>
          )}
        </div>
      ))}
    </div>
  );
}
