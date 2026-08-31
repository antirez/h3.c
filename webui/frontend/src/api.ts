import type {
  Asset,
  Capabilities,
  Invite,
  Job,
  JobSpec,
  SystemInfo,
  User,
  ValidationReport,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: init?.body ? { "Content-Type": "application/json" } : undefined,
    ...init,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new ApiError(response.status, detail?.detail ?? response.statusText);
  }
  return (await response.json()) as T;
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: unknown,
  ) {
    super(typeof detail === "string" ? detail : "request failed");
  }

  /** Validation failures arrive as { errors: [...] }. */
  get errors(): string[] {
    const detail = this.detail as { errors?: string[] } | string;
    if (typeof detail === "object" && Array.isArray(detail?.errors)) {
      return detail.errors;
    }
    return [String(this.message)];
  }
}

export const api = {
  system: () => request<SystemInfo>("/api/system"),
  capabilities: () => request<Capabilities>("/api/capabilities"),
  jobs: () => request<Job[]>("/api/jobs"),
  job: (id: number) => request<Job>(`/api/jobs/${id}`),
  validate: (spec: JobSpec) =>
    request<ValidationReport>("/api/jobs/validate", {
      method: "POST",
      body: JSON.stringify(spec),
    }),
  submit: (spec: JobSpec) =>
    request<Job>("/api/jobs", { method: "POST", body: JSON.stringify(spec) }),
  cancel: (id: number) =>
    request<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  /** Deletes the job and everything it wrote. Answers 204, with no body. */
  remove: async (id: number): Promise<void> => {
    const response = await fetch(`/api/jobs/${id}`, { method: "DELETE" });
    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      throw new ApiError(response.status, detail?.detail ?? response.statusText);
    }
  },
  assets: () => request<Asset[]>("/api/assets"),
  upload: async (file: File): Promise<Asset> => {
    const body = new FormData();
    body.append("file", file);
    const response = await fetch("/api/assets", { method: "POST", body });
    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      throw new ApiError(response.status, detail?.detail ?? response.statusText);
    }
    return (await response.json()) as Asset;
  },
  videoUrl: (id: number) => `/api/jobs/${id}/video`,
  posterUrl: (id: number) => `/api/jobs/${id}/poster`,
  logUrl: (id: number) => `/api/jobs/${id}/log`,
  previewUrl: (id: number, step: number | null) =>
    `/api/jobs/${id}/preview?step=${step ?? 0}`,
  assetUrl: (id: number) => `/api/assets/${id}/file`,

  me: () => request<User>("/api/auth/me"),
  login: (username: string, password: string) =>
    request<User>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),
  register: (username: string, password: string, invite: string) =>
    request<User>("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({ username, password, invite: invite || null }),
    }),
  logout: async (): Promise<void> => {
    await fetch("/api/auth/logout", { method: "POST" });
  },
  users: () => request<User[]>("/api/users"),
  invites: () => request<Invite[]>("/api/invites"),
  createInvite: () => request<{ code: string }>("/api/invites", { method: "POST" }),
  deleteUser: async (id: number): Promise<void> => {
    const response = await fetch(`/api/users/${id}`, { method: "DELETE" });
    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      throw new ApiError(response.status, detail?.detail ?? response.statusText);
    }
  },
  resetPassword: (id: number, password: string) =>
    request<User>(`/api/users/${id}/password`, {
      method: "POST",
      body: JSON.stringify({ password }),
    }),
};

/** Subscribe to one job's progress; returns an unsubscribe function. */
export function watchJob(id: number, onJob: (job: Job) => void): () => void {
  const source = new EventSource(`/api/jobs/${id}/events`);
  source.addEventListener("job", (event) => {
    onJob(JSON.parse((event as MessageEvent).data) as Job);
  });
  source.addEventListener("error", () => source.close());
  return () => source.close();
}
