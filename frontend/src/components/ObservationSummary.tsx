import { KPI_FIELDS } from "../config";
import type { Observation } from "../types";

export function ObservationSummary({
  observation,
}: {
  observation: Observation | null;
}) {
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
