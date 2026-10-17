type SiteSectionsProps = {
  onExample: (question: string) => void;
};

const examples = [
  {
    tag: "Troubleshoot",
    title: "Explain unexpectedly low throughput",
    copy: "Investigate signal quality, scheduling and the selected observation.",
    question: "Why might this measurement have this throughput?",
  },
  {
    tag: "Radio concepts",
    title: "Understand RSRP and SINR",
    copy: "Compare coverage measurements with interference and link quality.",
    question: "What is the difference between RSRP and SINR?",
  },
  {
    tag: "5G performance",
    title: "Connect CQI, MCS and data rates",
    copy: "Learn how channel quality affects modulation and coding.",
    question: "How do CQI and MCS affect data rate?",
  },
];

export function SiteSections({ onExample }: SiteSectionsProps) {
  return (
    <>
      <section className="editorial-section" id="examples" aria-labelledby="examples-title">
        <div className="editorial-heading">
          <span className="section-index">Explore</span>
          <h2 id="examples-title">Start with a real question.</h2>
          <p>Choose an example to prefill the assistant, or write your own.</p>
        </div>
        <div className="example-grid">
          {examples.map((example) => (
            <button
              className="example-card"
              key={example.title}
              type="button"
              onClick={() => onExample(example.question)}
            >
              <span className="example-tag">{example.tag}</span>
              <strong>{example.title}</strong>
              <span>{example.copy}</span>
              <span className="example-cta">Try this question ↗</span>
            </button>
          ))}
        </div>
      </section>

      <section className="editorial-section" id="data" aria-labelledby="data-title">
        <div className="editorial-heading">
          <span className="section-index">Reference data</span>
          <h2 id="data-title">Measurements and standards, together.</h2>
          <p>
            The assistant combines dataset-relative radio measurements with cited
            technical literature. Measurements provide context; retrieved standards
            help explain the mechanisms.
          </p>
        </div>
        <div className="proof-grid">
          <article className="proof-card">
            <span className="proof-number">01</span>
            <h3>Real-world KPI observations</h3>
            <p>
              Explore public Ericsson/AERPAW 5G NSA measurements, or type your own
              LTE and NR KPI values. Dataset statistics are descriptive, not
              universal performance thresholds.
            </p>
            <a href="https://doi.org/10.5061/dryad.wh70rxx06" target="_blank" rel="noopener noreferrer">
              Dataset and provenance ↗
            </a>
          </article>
          <article className="proof-card">
            <span className="proof-number">02</span>
            <h3>Technical sources you can inspect</h3>
            <p>
              Retrieval combines semantic and keyword matching over telecom
              documentation. Each answer includes source references and the
              retrieved evidence for you to inspect.
            </p>
            <a href="https://github.com/fredrik-nguyen-labs/Telecom-RAG/blob/main/docs/SOURCES.md" target="_blank" rel="noopener noreferrer">
              View source inventory ↗
            </a>
          </article>
          <article className="proof-card">
            <span className="proof-number">03</span>
            <h3>Bounded, transparent analysis</h3>
            <p>
              The system separates general questions from KPI-aware ones and
              computes deterministic statistics when relevant. Suggested causes
              are hypotheses, not proof of network faults.
            </p>
            <a href="#methodology">How analysis works ↓</a>
          </article>
        </div>
      </section>

      <section className="editorial-section" id="faq" aria-labelledby="faq-title">
        <div className="editorial-heading">
          <span className="section-index">Good to know</span>
          <h2 id="faq-title">Questions about the demo.</h2>
        </div>
        <div className="faq-stack">
          <details>
            <summary>Does every answer analyze the selected KPIs?</summary>
            <p>
              No. General telecom questions use documentation only. Questions
              about a selected observation can also use dataset-relative KPI analysis.
            </p>
          </details>
          <details>
            <summary>Are the answers definitive network diagnoses?</summary>
            <p>
              No. The assistant is an engineering demonstration. Measurements,
              reference datasets, retrieval and language models have limitations.
              Validate important conclusions against network traces and documentation.
            </p>
          </details>
          <details>
            <summary>Do you save my conversation?</summary>
            <p>
              This website does not store chat history in its application database
              or browser persistent storage. The active conversation stays in page
              memory and is sent for processing when you ask a question. Hosting
              and model providers may process technical metadata. Read the{" "}
              <a href="/privacy.html">privacy notice</a> for details.
            </p>
          </details>
          <details>
            <summary>Why can the first request take longer?</summary>
            <p>
              The demo API runs on an on-demand hosting tier that may need time
              to start after inactivity. If a request fails, use Retry question.
            </p>
          </details>
        </div>
      </section>
    </>
  );
}
