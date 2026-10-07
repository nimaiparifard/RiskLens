/**
 * Fetch wrapper: JWT bearer auth, transparent refresh-token rotation on 401,
 * typed errors, and Zod validation of every response body.
 */
import type { z } from "zod";

const BASE = import.meta.env.VITE_API_URL ?? "";
const ACCESS = "risklens.access";
const REFRESH = "risklens.refresh";

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public body: unknown = null,
  ) {
    super(message);
  }
}

export const tokens = {
  get access() {
    return localStorage.getItem(ACCESS);
  },
  get refresh() {
    return localStorage.getItem(REFRESH);
  },
  set(access: string, refresh?: string) {
    localStorage.setItem(ACCESS, access);
    if (refresh) localStorage.setItem(REFRESH, refresh);
  },
  clear() {
    localStorage.removeItem(ACCESS);
    localStorage.removeItem(REFRESH);
  },
};

let onUnauthorized: () => void = () => {};
export function setUnauthorizedHandler(fn: () => void) {
  onUnauthorized = fn;
}

let refreshing: Promise<boolean> | null = null;
async function refreshAccess(): Promise<boolean> {
  const refresh = tokens.refresh;
  if (!refresh) return false;
  refreshing ??= fetch(`${BASE}/api/auth/token/refresh/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh }),
  })
    .then(async (r) => {
      if (!r.ok) return false;
      const data = await r.json();
      tokens.set(data.access, data.refresh);
      return true;
    })
    .catch(() => false)
    .finally(() => {
      refreshing = null;
    });
  return refreshing;
}

function errorMessage(body: unknown, fallback: string): string {
  if (body && typeof body === "object") {
    const b = body as Record<string, unknown>;
    if (typeof b.detail === "string") return b.detail;
    const first = Object.entries(b).find(([k]) => k !== "error");
    if (first) return `${first[0]}: ${Array.isArray(first[1]) ? first[1].join(" ") : JSON.stringify(first[1])}`;
  }
  return fallback;
}

export async function request<T extends z.ZodTypeAny>(
  path: string,
  schema: T,
  init: RequestInit & { json?: unknown } = {},
  retry = true,
): Promise<z.infer<T>> {
  const headers = new Headers(init.headers);
  if (init.json !== undefined) headers.set("Content-Type", "application/json");
  if (tokens.access) headers.set("Authorization", `Bearer ${tokens.access}`);
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers,
    body: init.json !== undefined ? JSON.stringify(init.json) : init.body,
  });
  if (res.status === 401 && retry && tokens.refresh) {
    if (await refreshAccess()) return request(path, schema, init, false);
    tokens.clear();
    onUnauthorized();
  }
  const text = await res.text();
  const body = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const code = body && typeof body === "object" && "error" in body ? String(body.error) : `http_${res.status}`;
    throw new ApiError(res.status, code, errorMessage(body, res.statusText), body);
  }
  const parsed = schema.safeParse(body);
  if (!parsed.success) {
    console.error("API contract violation", path, parsed.error.issues);
    throw new ApiError(500, "contract_violation", `Unexpected response from ${path}`, parsed.error.issues);
  }
  return parsed.data;
}
