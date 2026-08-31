import { useRef, useState } from "react";

import { api } from "../api";
import { REFERENCE_RULES } from "../generated/options";
import { resolvedFrames } from "../spec";
import type { Asset, JobSpec, ReferenceKind } from "../types";

interface Props {
  spec: JobSpec;
  onChange: (spec: JobSpec) => void;
  onUploaded: (asset: Asset) => void;
}

const KINDS: { kind: ReferenceKind; label: string; flag: string; short: string }[] = [
  { kind: "image", label: "photo", flag: "--ref-image", short: "photo" },
  { kind: "video", label: "clip", flag: "--ref-video", short: "clip" },
  { kind: "silent_video", label: "clip, no sound", flag: "--ref-silent-video", short: "clip" },
  { kind: "video_audio", label: "clip + sound", flag: "--ref-video-audio", short: "clip+snd" },
  { kind: "audio", label: "sound", flag: "--ref-audio", short: "sound" },
];

/** Reference material, in the order h3 will read it. */
export function References({ spec, onChange, onUploaded }: Props) {
  const [pending, setPending] = useState<ReferenceKind | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const references = spec.references;
  const videoKinds: ReferenceKind[] = ["video", "silent_video", "video_audio"];
  const counts = {
    images: references.filter((r) => r.kind === "image").length,
    videos: references.filter((r) => videoKinds.includes(r.kind)).length,
    audio: references.filter((r) => ["audio", "video", "video_audio"].includes(r.kind)).length,
    seconds: references
      .filter((r) => r.kind === "audio")
      .reduce((total, r) => total + (r.seconds ?? 0), 0),
  };
  const full = references.length >= REFERENCE_RULES.max_total;

  function move(index: number, delta: number) {
    const next = [...references];
    const target = index + delta;
    if (target < 0 || target >= next.length) return;
    [next[index], next[target]] = [next[target], next[index]];
    onChange({ ...spec, references: next });
  }

  async function accept(file: File | undefined) {
    if (!file || !pending) return;
    try {
      const asset = await api.upload(file);
      onUploaded(asset);
      onChange({
        ...spec,
        references: [
          ...references,
          {
            kind: pending,
            path: asset.path,
            label: asset.filename,
            seconds: asset.metadata.seconds ?? null,
          },
        ],
      });
    } finally {
      setPending(null);
    }
  }

  return (
    <div className="inner">
      <p className="why" style={{ marginTop: 0 }}>
        Material to draw from: a photo for the subject, a clip to continue, a sound for the
        mood. The order matters, and reference material cannot be combined with a start or
        end photo.
      </p>

      {references.length === 0 ? (
        <div className="empty">Nothing added yet.</div>
      ) : (
        <div className="reflist">
          {references.map((reference, index) => (
            <div className="ref" key={`${reference.path}-${index}`}>
              <span className="k">
                {KINDS.find((entry) => entry.kind === reference.kind)?.short}
              </span>
              <span className="n">{reference.label ?? reference.path.split("/").pop()}</span>
              <span className="m">
                {reference.seconds ? `${reference.seconds.toFixed(1)} s · ` : ""}
                {KINDS.find((entry) => entry.kind === reference.kind)?.flag}
              </span>
              <span className="btns">
                <button onClick={() => move(index, -1)} aria-label="move up">↑</button>
                <button onClick={() => move(index, 1)} aria-label="move down">↓</button>
                <button
                  aria-label="remove"
                  onClick={() =>
                    onChange({
                      ...spec,
                      references: references.filter((_, at) => at !== index),
                    })
                  }
                >
                  ✕
                </button>
              </span>
            </div>
          ))}
        </div>
      )}

      <div className="counters">
        <span>
          in all <b>{references.length}</b>/{REFERENCE_RULES.max_total}
        </span>
        <span>
          photos <b>{counts.images}</b>/{REFERENCE_RULES.max_images}
        </span>
        <span>
          clips <b>{counts.videos}</b>/{REFERENCE_RULES.max_videos}
        </span>
        <span>
          sounds <b>{counts.audio}</b>/{REFERENCE_RULES.max_audio_inputs}
        </span>
        <span>
          sound length <b>{counts.seconds.toFixed(1)}</b>/15 s
        </span>
      </div>

      <div className="tries" style={{ marginTop: 12 }}>
        <em>Add:</em>
        {KINDS.map((entry) => (
          <button
            key={entry.kind}
            className="try"
            disabled={full}
            onClick={() => {
              setPending(entry.kind);
              input.current?.click();
            }}
          >
            {entry.label}
          </button>
        ))}
      </div>
      <input
        ref={input}
        type="file"
        hidden
        onChange={(event) => void accept(event.target.files?.[0])}
      />

      <ul className="rules" style={{ marginTop: 10 }}>
        <li>
          At most {REFERENCE_RULES.max_total} items: {REFERENCE_RULES.max_images} photos,{" "}
          {REFERENCE_RULES.max_videos} clips, {REFERENCE_RULES.max_audio_inputs} sounds.
        </li>
        <li>A sound needs a photo or a clip beside it, and must last 2 to 15 seconds.</li>
        <li>
          A clip's own sound is trimmed to the video's length and needs 2 seconds, so ask
          for at least 56 frames — this video is {resolvedFrames(spec)}.
        </li>
      </ul>
    </div>
  );
}
