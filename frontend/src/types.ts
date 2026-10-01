export type Source = {
  citation: string;
  source?: string;
  source_id?: string;
  title?: string;
  url?: string;
  page?: number | null;
  section?: string | null;
  content?: string;
  excerpt?: string;
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

export type Observation = Record<string, string | number | boolean | null | undefined>;

export type ChatResponse = {
  answer: string;
  sources: Source[];
  route?: string;
  route_reason?: string;
  latency_s?: number;
  rate_limit_remaining?: number;
};

export type HealthResponse = {
  status: string;
  documents: number;
  observations: number;
  generator: string;
  router: string;
};
