import type {
  ChatMessage,
  ChatResponse,
  HealthResponse,
  Observation,
} from "./types";

// The API's Render hostname is the stable production origin. The optional
// custom API subdomain is not required for the public frontend to function.
const DEFAULT_API_BASE = import.meta.env.DEV
  ? "http://localhost:8000"
  : "https://telecom-rag-api.onrender.com";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE).replace(
  /\/$/,
  "",
);

const delay = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const isRead = !init?.method || init.method.toUpperCase() === "GET";
  const attempts = isRead ? 2 : 1;

  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      // On a free Render instance, the first request after an idle period
      // may need additional time to wake the backend.
      const response = await fetch(`${API_BASE}${path}`, {
        ...init,
        signal: AbortSignal.timeout(isRead ? 75_000 : 120_000),
      });
      const body = await response.json().catch(() => ({}));

      if (!response.ok) {
        if (isRead && attempt < attempts - 1 && [502, 503, 504].includes(response.status)) {
          await delay(1_200);
          continue;
        }
        const detail =
          typeof body.detail === "string"
            ? body.detail
            : `API returned HTTP ${response.status}.`;
        throw new Error(detail);
      }
      return body as T;
    } catch (error) {
      if (error instanceof TypeError) {
        if (isRead && attempt < attempts - 1) {
          await delay(1_200);
          continue;
        }
        throw new Error(
          "Network connection to the API failed. Please try again. " +
            "The free backend may need time to wake up.",
        );
      }
      if (error instanceof DOMException && error.name === "TimeoutError") {
        throw new Error(
          "The API request timed out. The server may still be processing it. Please retry.",
        );
      }
      throw error;
    }
  }

  throw new Error("Could not reach the API. Please try again.");
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
  // Do not automatically retry POST: the original generation may have finished
  // on the server even if its response failed to reach the browser.
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
