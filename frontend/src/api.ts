import type {
  ChatMessage,
  ChatResponse,
  HealthResponse,
  Observation,
} from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(
  /\/$/,
  "",
);

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.detail || `Request failed with status ${response.status}`);
  }
  return body as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

export async function getObservations(limit = 40): Promise<Observation[]> {
  const response = await request<{ items: Observation[] }>(
    `/api/observations?limit=${limit}`,
  );
  return response.items;
}

export function sendChat(
  question: string,
  observation: Observation | null,
  history: ChatMessage[],
): Promise<ChatResponse> {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      observation,
      history: history.slice(-8).map(({ role, content }) => ({ role, content })),
    }),
  });
}
