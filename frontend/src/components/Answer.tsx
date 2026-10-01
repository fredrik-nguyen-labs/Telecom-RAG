import type { Source } from "../types";

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

export function AnswerBody({ text }: { text: string }) {
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

export function SourceList({
  answer,
  sources,
}: {
  answer: string;
  sources: Source[];
}) {
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
