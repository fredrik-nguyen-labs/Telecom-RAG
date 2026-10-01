import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { getHealth, getObservations, sendChat } from "./api";
import type {
  ChatMessage,
  HealthResponse,
  Observation,
  Source,
} from "./types";

const DEFAULT_QUESTION =
  "Why might this measurement have this throughput, and which radio measurements are most relevant to investigate?";

const SUGGESTIONS = [
  "Why might this measurement have this throughput?",
  "Which radio measurements should I investigate first?",
  "What stands out about this measurement?",
  "What is the difference between RSRP and SINR?",
  "How do CQI and MCS affect data rate?",
  "What is beamforming in 5G and why is it useful?",
  "How does 5G NSA differ from standalone 5G?",
  "What does rank indicator mean in MIMO?",
  "Why can throughput be low even with strong signal strength?",
];

const KPI_FIELDS = [
  { key: "nr_rsrp_dbm", label: "NR RSRP", unit: "dBm", initial: "-95" },
  { key: "nr_sinr_db", label: "NR SINR", unit: "dB", initial: "10" },
  { key: "nr_cqi", label: "NR CQI", unit: "", initial: "12" },
  { key: "nr_ri", label: "NR RI", unit: "", initial: "2" },
  { key: "lte_rsrp_dbm", label: "LTE RSRP", unit: "dBm", initial: "-90" },
  { key: "lte_sinr_db", label: "LTE SINR", unit: "dB", initial: "15" },
  { key: "nr_mcs", label: "NR MCS", unit: "", initial: "18" },
  { key: "throughput_mbps", label: "Throughput", unit: "Mbps", initial: "50" },
] as const;

type KpiKey = (typeof KPI_FIELDS)[number]["key"];

const DEFAULT_KPIS = Object.fromEntries(
  KPI_FIELDS.map((field) => [field.key, field.initial]),
) as Record<KpiKey, string>;

function sourceLink(source: Source): string | undefined {
  if (!source.url) return undefined;
  if (source.page && source.url.toLowerCase().endsWith(".pdf")) {
    return `${source.url}#page=${source.page}`;
  }
  return source.url;
}

function citationClaims(answer: string): Map<string, string[]> {
  const claims = new Map<string, string[]>();
  const spans = answer.match(/[^.!?\n]+[.!?]?/g) ?? [];

  for (const rawSpan of spans) {
    const span = rawSpan.trim();
    if (!span) continue;
    const citations = [...span.matchAll(/\[S(\d+)\]/g)].map(
      (match) => `S${match[1]}`,
    );
    if (!citations.length) continue;

    const claim = span.replace(/\s*\[S\d+\]/g, "").trim();
    for (const citation of citations) {
      const existing = claims.get(citation) ?? [];
      if (claim && !existing.includes(claim)) existing.push(claim);
      claims.set(citation, existing);
    }
  }

  return claims;
}

function InlineAnswer({ text }: { text: string }) {
  const parts = text.split(/(\[S\d+\])/g);
  return (
    <>
      {parts.map((part, index) => {
        const match = part.match(/^\[S(\d+)\]$/);
        if (!match) return <span key={index}>{part}</span>;
        return (
          <a className="citation" href={`#source-S${match[1]}`} key={index}>
            {part}
          </a>
        );
      })}
    </>
  );
}

function AnswerBody({ text }: { text: string }) {
  const groups = text.split(/\n{2,}/).filter((group) => group.trim());

  return (
    <div className="answer-body">
      {groups.map((group, index) => {
        const lines = group
          .split("\n")
          .map((line) => line.trim())
          .filter(Boolean);
        const isList = lines.length > 0 && lines.every((line) => line.startsWith("- "));
        if (isList) {
          return (
            <ul key={index}>
              {lines.map((line, lineIndex) => (
                <li key={lineIndex}>
                  <InlineAnswer text={line.slice(2)} />
                </li>
              ))}
            </ul>
          );
        }

        return (
          <p key={index}>
            {lines.map((line, lineIndex) => (
              <span key={lineIndex}>
                {lineIndex > 0 && <br />}
                <InlineAnswer text={line} />
              </span>
            ))}
          </p>
        );
      })}
    </div>
  );
}

function SourceList({ answer, sources }: { answer: string; sources: Source[] }) {
  if (!sources.length) return null;
  const claims = citationClaims(answer);

  return (
    <div className="sources">
      <div className="sources-title">Sources</div>
      {sources.map((source, index) => {
        const citation = source.citation || `S${index + 1}`;
        const title = source.title || source.source || "Unknown source";
        const url = sourceLink(source);
        const location = [
          source.page ? `p. ${source.page}` : "",
          source.section || "",
        ]
          .filter(Boolean)
          .join(" · ");
        const supportedClaims = claims.get(citation) ?? [];

        return (
          <article className="source-card" id={`source-${citation}`} key={citation}>
            <div className="source-reference">
              <span className="reference-chip">[{citation}]</span>
              {location && <span>{location}</span>}
            </div>

            {supportedClaims.length ? (
              supportedClaims.map((claim) => (
                <p className="source-row" key={claim}>
                  <strong>Claim:</strong> {claim}
                </p>
              ))
            ) : (
              <p className="source-row">
                <strong>Claim:</strong> Retrieved reference not cited in the answer.
              </p>
            )}

            <p className="source-row">
              <strong>Source:</strong>{" "}
              {url ? (
                <a href={url} rel="noreferrer" target="_blank">
                  {title}
                </a>
              ) : (
                title
              )}
            </p>

            {url && (
              <p className="source-row source-url">
                <a href={url} rel="noreferrer" target="_blank">
                  {url}
                </a>
              </p>
            )}

            <details>
              <summary>Retrieved evidence</summary>
              <div className="evidence">
                {source.content || source.excerpt || "No chunk text available."}
              </div>
            </details>
          </article>
        );
      })}
    </div>
  );
}

function ObservationSummary({ observation }: { observation: Observation | null }) {
  if (!observation) return null;

  const rows = KPI_FIELDS.flatMap((field) => {
    const value = observation[field.key];
    if (value === null || value === undefined || value === "") return [];
    return [{ ...field, value }];
  });

  const anomaly = observation.anomaly_score;

  return (
    <div className="observation-summary">
      {rows.map((row) => (
        <div className="metric" key={row.key}>
          <span>{row.label}</span>
          <strong>
            {String(row.value)}
            {row.unit ? ` ${row.unit}` : ""}
          </strong>
        </div>
      ))}
      {anomaly !== null && anomaly !== undefined && (
        <div className="metric">
          <span>Anomaly score</span>
          <strong>{Number(anomaly).toFixed(3)}</strong>
        </div>
      )}
    </div>
  );
}

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState(DEFAULT_QUESTION);
  const [mode, setMode] = useState<"custom" | "dataset">("custom");
  const [customKpis, setCustomKpis] = useState(DEFAULT_KPIS);
  const [observations, setObservations] = useState<Observation[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const conversationEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((err: Error) => setHealthError(err.message));

    getObservations()
      .then((items) => {
        setObservations(items);
        const first = items[0]?.observation_id;
        if (first !== null && first !== undefined) setSelectedId(String(first));
      })
      .catch(() => {
        // Custom KPI mode remains fully usable if sample loading is unavailable.
      });
  }, []);

  useEffect(() => {
    conversationEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const customObservation = useMemo<Observation | null>(() => {
    const output: Observation = {
      observation_id: "custom",
      observation_source: "user-entered",
    };
    let hasValue = false;

    for (const field of KPI_FIELDS) {
      const raw = customKpis[field.key].trim();
      if (!raw) continue;
      const value = Number(raw);
      if (Number.isFinite(value)) {
        output[field.key] = value;
        hasValue = true;
      }
    }
    return hasValue ? output : null;
  }, [customKpis]);

  const datasetObservation = useMemo(
    () =>
      observations.find(
        (observation) => String(observation.observation_id) === selectedId,
      ) ?? null,
    [observations, selectedId],
  );

  const activeObservation = mode === "custom" ? customObservation : datasetObservation;

  async function submit(event?: FormEvent) {
    event?.preventDefault();
    const cleaned = question.trim();
    if (!cleaned || loading) return;

    const history = messages;
    setMessages((current) => [...current, { role: "user", content: cleaned }]);
    setQuestion("");
    setLoading(true);
    setError("");

    try {
      const result = await sendChat(cleaned, activeObservation, history);
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: result.answer,
          sources: result.sources,
        },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "The request failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/">
          <span className="brand-mark">FN</span>
          <span>
            <strong>Telecom RAG</strong>
            <small>5G network diagnostics</small>
          </span>
        </a>
        <nav>
          <a
            href="https://github.com/fredrik-nguyen-labs/Telecom-RAG"
            rel="noreferrer"
            target="_blank"
          >
            GitHub
          </a>
          <a href="#about">About</a>
        </nav>
      </header>

      <main>
        <section className="hero">
          <div className="eyebrow">Grounded network analysis</div>
          <h1>Diagnose 5G measurements with data and technical evidence.</h1>
          <p>
            Explore real Ericsson/AERPAW radio measurements, compare your own KPIs
            against the dataset, and ask telecom questions grounded in standards and
            engineering documentation.
          </p>
          <div className="hero-meta">
            <span className={health ? "status-dot online" : "status-dot"} />
            {health ? (
              <span>
                API online · {health.documents.toLocaleString()} chunks ·{" "}
                {health.observations.toLocaleString()} KPI observations
              </span>
            ) : healthError ? (
              <span>API unavailable · {healthError}</span>
            ) : (
              <span>Connecting to API…</span>
            )}
          </div>
        </section>

        <section className="workspace">
          <div className="section-heading">
            <div>
              <span className="section-index">01</span>
              <h2>Conversation</h2>
            </div>
            {messages.length > 0 && (
              <button className="ghost-button" onClick={() => setMessages([])}>
                New chat
              </button>
            )}
          </div>

          <div className="conversation">
            {!messages.length && !loading && (
              <div className="empty-state">
                <div className="empty-icon">↗</div>
                <strong>Ask about a measurement or a telecom concept.</strong>
                <span>
                  Recent turns stay in context for follow-up questions. Your KPI
                  selection stays separate from the conversation.
                </span>
              </div>
            )}

            {messages.map((message, index) => (
              <div className={`message ${message.role}`} key={index}>
                <div className="message-label">
                  {message.role === "user" ? "You" : "Telecom RAG"}
                </div>
                <AnswerBody text={message.content} />
                {message.role === "assistant" && (
                  <SourceList
                    answer={message.content}
                    sources={message.sources ?? []}
                  />
                )}
              </div>
            ))}

            {loading && (
              <div className="message assistant loading-card">
                <div className="message-label">Telecom RAG</div>
                <div className="loading-line">
                  <span className="pulse" />
                  Routing, retrieving evidence, and generating a grounded answer…
                </div>
              </div>
            )}
            <div ref={conversationEnd} />
          </div>

          {error && <div className="error-banner">{error}</div>}
        </section>

        <section className="control-grid">
          <article className="panel">
            <div className="panel-heading">
              <span className="section-index">02</span>
              <div>
                <h2>KPI context</h2>
                <p>Use your own measurement or a real dataset observation.</p>
              </div>
            </div>

            <div className="segmented">
              <button
                className={mode === "custom" ? "active" : ""}
                onClick={() => setMode("custom")}
              >
                Custom KPIs
              </button>
              <button
                className={mode === "dataset" ? "active" : ""}
                onClick={() => setMode("dataset")}
              >
                Dataset sample
              </button>
            </div>

            {mode === "custom" ? (
              <>
                <div className="kpi-form">
                  {KPI_FIELDS.map((field) => (
                    <label key={field.key}>
                      <span>
                        {field.label}
                        {field.unit && <small>{field.unit}</small>}
                      </span>
                      <input
                        inputMode="decimal"
                        value={customKpis[field.key]}
                        onChange={(event) =>
                          setCustomKpis((current) => ({
                            ...current,
                            [field.key]: event.target.value,
                          }))
                        }
                      />
                    </label>
                  ))}
                </div>
                <p className="panel-note">
                  Unknown values can be left blank. Dataset-relative statistics are
                  only used when the question actually needs KPI analysis.
                </p>
              </>
            ) : (
              <>
                <label className="dataset-select">
                  <span>Observation</span>
                  <select
                    value={selectedId}
                    onChange={(event) => setSelectedId(event.target.value)}
                  >
                    {observations.map((observation) => (
                      <option
                        key={String(observation.observation_id)}
                        value={String(observation.observation_id)}
                      >
                        {String(observation.observation_id)}
                        {observation.anomaly_score !== null &&
                        observation.anomaly_score !== undefined
                          ? ` · anomaly ${Number(observation.anomaly_score).toFixed(3)}`
                          : ""}
                      </option>
                    ))}
                  </select>
                </label>
                <ObservationSummary observation={datasetObservation} />
              </>
            )}
          </article>

          <article className="panel composer-panel">
            <div className="panel-heading">
              <span className="section-index">03</span>
              <div>
                <h2>Ask a question</h2>
                <p>The router decides whether the selected KPI context is relevant.</p>
              </div>
            </div>

            <label className="suggestion-select">
              <span>Suggested question</span>
              <select
                defaultValue=""
                onChange={(event) => {
                  if (event.target.value) setQuestion(event.target.value);
                }}
              >
                <option value="">Choose a starting point…</option>
                {SUGGESTIONS.map((suggestion) => (
                  <option key={suggestion} value={suggestion}>
                    {suggestion}
                  </option>
                ))}
              </select>
            </label>

            <form onSubmit={submit}>
              <textarea
                maxLength={700}
                placeholder="Ask about the selected measurement or a general telecom topic…"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
              />
              <div className="composer-footer">
                <span>{question.length}/700</span>
                <button
                  className="primary-button"
                  disabled={!question.trim() || loading}
                  type="submit"
                >
                  {loading ? "Analyzing…" : "Send question"}
                </button>
              </div>
            </form>

            <ObservationSummary observation={activeObservation} />
          </article>
        </section>

        <section className="about" id="about">
          <div>
            <span className="section-index">04</span>
            <h2>How the answer is produced</h2>
          </div>
          <div className="pipeline">
            <div>
              <strong>Route</strong>
              <span>A lightweight semantic router chooses docs-only or KPI + docs.</span>
            </div>
            <div>
              <strong>Analyze</strong>
              <span>KPI-aware questions get deterministic dataset-relative context.</span>
            </div>
            <div>
              <strong>Retrieve</strong>
              <span>Supabase pgvector + FTS + RRF, then BGE reranking.</span>
            </div>
            <div>
              <strong>Ground</strong>
              <span>Gemma answers from the retrieved technical evidence.</span>
            </div>
          </div>
        </section>
      </main>

      <footer>
        <div>
          <strong>Telecom RAG</strong>
          <span>Research and engineering demo by Fredrik Nguyen.</span>
        </div>
        <span>
          React · FastAPI · LangGraph · Supabase · Cloudflare Workers AI
        </span>
      </footer>
    </div>
  );
}

export default App;
