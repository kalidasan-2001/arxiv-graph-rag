/// <reference types="vite/client" />

function defaultApiBaseUrl(): string {
  if (typeof window === "undefined") {
    return "http://localhost:8000";
  }
  return `${window.location.protocol}//${window.location.hostname}:8000`;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
    readonly detail?: unknown
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export function apiBaseUrl(): string {
  return (import.meta.env.VITE_API_BASE_URL || defaultApiBaseUrl()).replace(/\/$/, "");
}

export async function postJson<TResponse>(path: string, body: unknown): Promise<TResponse> {
  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl()}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
  } catch (error) {
    throw new ApiError("Backend is unreachable. Check that FastAPI is running on the configured API URL.", undefined, error);
  }

  const payload = await readJson(response);
  if (!response.ok) {
    const detail = payload && typeof payload === "object" && "detail" in payload ? payload.detail : payload;
    throw new ApiError(`Backend returned ${response.status}: ${formatErrorDetail(detail)}`, response.status, detail);
  }
  return payload as TResponse;
}

function formatErrorDetail(detail: unknown): string {
  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }
  if (detail == null) {
    return "No error detail returned.";
  }
  try {
    return JSON.stringify(detail);
  } catch {
    return "Unprintable error detail returned.";
  }
}

async function readJson(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch {
    throw new ApiError("Backend returned a non-JSON response.", response.status, text);
  }
}
