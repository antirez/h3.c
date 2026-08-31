export type ReferenceKind =
  | "image"
  | "video"
  | "silent_video"
  | "video_audio"
  | "audio";

export interface User {
  id?: number;
  username: string;
  role: "admin" | "user";
  created_at?: string;
}

export interface Invite {
  code: string;
  created_at: string;
  used: boolean;
}

export interface Reference {
  kind: ReferenceKind;
  path: string;
  audio_path?: string | null;
  seconds?: number | null;
  /** Local only: what to show in the list. */
  label?: string;
}

export interface JobSpec {
  prompt: string;
  width: number;
  height: number;
  render_width: number;
  render_height: number;
  frames: number | null;
  seconds: number | null;
  steps: number;
  denoise_reuse: number;
  dit_layers: number;
  core_reuse: number;
  token_reduction: boolean;
  ssd_streaming: boolean;
  use_int8_row_fc2: boolean;
  use_reference_rope: boolean;
  seed: number;
  first_frame: string | null;
  last_frame: string | null;
  references: Reference[];
  reference_image_size: "match" | "max";
  write_frames: boolean;
  profile: boolean;
  preview: boolean;
  slower: string[];
  postprocess: string[];
}

export type JobState =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface Job {
  id: number;
  state: JobState;
  prompt: string;
  params: JobSpec;
  argv: string[] | null;
  phase: string | null;
  completed: number;
  total: number;
  progress: number;
  error: string | null;
  output_path: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  elapsed: number | null;
  remaining: number | null;
  preview_step: number | null;
  warnings?: string[];
}

export interface Asset {
  id: number;
  sha256: string;
  kind: "image" | "video" | "audio";
  filename: string;
  path: string;
  bytes: number;
  metadata: {
    seconds?: number | null;
    width?: number | null;
    height?: number | null;
    has_audio?: boolean;
    notes?: string[];
  };
  duplicate?: boolean;
}

export interface SystemInfo {
  available: boolean;
  reason?: string;
  engine?: string;
  version?: string;
  device: Record<string, string>;
  components: Record<string, { files: number; tensors: number; gib: number }>;
  has_ref2va?: boolean;
}

export interface Plugin {
  name: string;
  label: string;
  description: string;
  env_var: string;
  available: boolean;
  reason: string | null;
  notice: string | null;
}

export interface Capabilities {
  plugins: Plugin[];
}

export interface ValidationReport {
  errors: string[];
  warnings: string[];
  frames: number;
  seconds: number;
}
