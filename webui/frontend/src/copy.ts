import copy from "../../shared/copy.json";

interface OptionCopy {
  name: string;
  help?: string;
  time?: string;
}

interface ErrorCopy {
  match: string;
  title: string;
  fix: string;
}

const OPTIONS = copy.options as Record<string, OptionCopy>;
const PHASES = copy.phases as Record<string, string>;
const STATES = copy.states as Record<string, string>;
const ERRORS = copy.errors as ErrorCopy[];

/** What a person calls this setting. */
export function optionName(key: string): string {
  return OPTIONS[key]?.name ?? key;
}

export function optionHelp(key: string): string | undefined {
  return OPTIONS[key]?.help;
}

/** What h3 is doing, said in words anyone can read. */
export function phaseName(phase: string | null): string {
  if (!phase) return "Getting started";
  return PHASES[phase] ?? phase;
}

export function stateName(state: string): string {
  return STATES[state] ?? state;
}

/** Turn an engine message into something that says what to change. */
export function explain(message: string): ErrorCopy {
  const found = ERRORS.find((entry) => message.includes(entry.match));
  return (
    found ?? {
      match: message,
      title: "That job cannot be made as it is.",
      fix: "Change one of the settings above and try again.",
    }
  );
}

/** "4 min", "1 h 12 min", "40 s" — the way a person reads a wait. */
export function humanMinutes(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !isFinite(seconds)) return "—";
  if (seconds < 90) return `${Math.max(1, Math.round(seconds))} s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min`;
  return `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

/** 00:04.5 — the frame counter's voice, for a length. */
export function timecode(seconds: number): string {
  const whole = Math.floor(seconds);
  const tenths = Math.round((seconds - whole) * 10);
  return `00:${String(whole).padStart(2, "0")}.${tenths}`;
}

/** 00:03:41 — for elapsed and remaining. */
export function clock(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "--:--";
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = total % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return hours ? `${pad(hours)}:${pad(minutes)}:${pad(rest)}` : `${pad(minutes)}:${pad(rest)}`;
}

/** The phases h3 walks through, in order, for the perforation rail. */
export const PHASE_ORDER = [
  "tokenizer",
  "text encoder",
  "Qwen vision",
  "refine text",
  "precompute AdaLN",
  "load transformer core",
  "preview VAE load",
  "denoise",
  "audio VAE encoder",
  "video VAE encoder",
  "audio VAE",
  "video VAE load",
  "FFmpeg",
];

/** Which mark of the rail a reported phase belongs to.
 *
 *  h3 reports sub-phases the rail has no mark for — `denoise enqueue` while
 *  streaming weights from SSD, for one. Falling back to the longest listed
 *  phase the report starts with keeps the rail on `denoise` instead of
 *  blanking it for the whole of the longest stretch of the job.
 */
export function railIndex(phases: string[], phase: string | null | undefined): number {
  if (!phase) return -1;
  const exact = phases.indexOf(phase);
  if (exact >= 0) return exact;
  let best = -1;
  phases.forEach((listed, index) => {
    if (phase.startsWith(listed) && (best < 0 || listed.length > phases[best].length)) {
      best = index;
    }
  });
  return best;
}

/** The phases this particular job will pass through.
 *
 *  A rail with marks that can never fill is a rail that lies: reference
 *  material adds encoding phases, and the preview adds a load of its own.
 */
export function railPhases(job: {
  params: { preview?: boolean; references?: unknown[] };
}): string[] {
  const withReferences = (job.params.references?.length ?? 0) > 0;
  return PHASE_ORDER.filter((phase) => {
    if (phase === "preview VAE load") return Boolean(job.params.preview);
    if (phase === "Qwen vision") return withReferences;
    if (phase.endsWith("VAE encoder")) return withReferences;
    return true;
  });
}
