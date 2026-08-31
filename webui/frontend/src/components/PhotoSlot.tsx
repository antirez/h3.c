import { useRef, useState } from "react";

import { api } from "../api";
import type { Asset } from "../types";

interface Props {
  title: string;
  subtitle: string;
  assets: Asset[];
  value: string | null;
  disabled?: boolean;
  onPick: (asset: Asset | null) => void;
  onUploaded: (asset: Asset) => void;
}

/** A drop target that doubles as a picker over photos already uploaded. */
export function PhotoSlot(props: Props) {
  const { title, subtitle, assets, value, disabled, onPick, onUploaded } = props;
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [browsing, setBrowsing] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const chosen = assets.find((asset) => asset.path === value) ?? null;
  const photos = assets.filter((asset) => asset.kind === "image");

  async function accept(file: File | undefined) {
    if (!file) return;
    setBusy(true);
    setProblem(null);
    try {
      const asset = await api.upload(file);
      onUploaded(asset);
      onPick(asset);
    } catch (failure) {
      setProblem(failure instanceof Error ? failure.message : "That file did not load.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <button
        className={`slot${chosen ? " filled" : ""}${over ? " drop-over" : ""}`}
        disabled={disabled}
        onClick={() => (chosen ? setBrowsing((open) => !open) : input.current?.click())}
        onDragOver={(event) => {
          event.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(event) => {
          event.preventDefault();
          setOver(false);
          if (!disabled) void accept(event.dataTransfer.files[0]);
        }}
      >
        <span className="plus">{chosen ? "▣" : "+"}</span>
        <span className="t">
          {busy ? "Loading…" : chosen ? chosen.filename : title}
          <span className="s">{chosen ? "click to change" : subtitle}</span>
        </span>
        {chosen ? (
          <span
            className="clear"
            role="button"
            onClick={(event) => {
              event.stopPropagation();
              onPick(null);
            }}
          >
            remove
          </span>
        ) : null}
      </button>
      <input
        ref={input}
        type="file"
        hidden
        accept="image/*"
        onChange={(event) => void accept(event.target.files?.[0])}
      />
      {!disabled && photos.length > 0 && !chosen ? (
        <button className="lib" style={{ marginTop: 7 }} onClick={() => setBrowsing((o) => !o)}>
          {browsing ? "hide photos" : "use a photo you already added"}
        </button>
      ) : null}
      {browsing ? (
        <div className="libgrid">
          {photos.map((asset) => (
            <button
              key={asset.id}
              className={`libpick${asset.path === value ? " on" : ""}`}
              onClick={() => {
                onPick(asset);
                setBrowsing(false);
              }}
            >
              <img className="thumb" src={api.assetUrl(asset.id)} alt="" />
              <span className="cap">{asset.filename}</span>
            </button>
          ))}
          <button className="libpick" onClick={() => input.current?.click()}>
            <span className="thumb" style={{ display: "grid", placeItems: "center" }}>
              +
            </span>
            <span className="cap">add another</span>
          </button>
        </div>
      ) : null}
      {problem ? <div className="note" style={{ color: "var(--fail)" }}>{problem}</div> : null}
    </div>
  );
}
