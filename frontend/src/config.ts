export const DEFAULT_QUESTION =
  "Why might this measurement have this throughput, and which radio measurements are most relevant to investigate?";

export const SUGGESTIONS = [
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

export const KPI_FIELDS = [
  { key: "nr_rsrp_dbm", label: "NR RSRP", unit: "dBm", initial: "-95" },
  { key: "nr_sinr_db", label: "NR SINR", unit: "dB", initial: "10" },
  { key: "nr_cqi", label: "NR CQI", unit: "", initial: "12" },
  { key: "nr_ri", label: "NR RI", unit: "", initial: "2" },
  { key: "lte_rsrp_dbm", label: "LTE RSRP", unit: "dBm", initial: "-90" },
  { key: "lte_sinr_db", label: "LTE SINR", unit: "dB", initial: "15" },
  { key: "nr_mcs", label: "NR MCS", unit: "", initial: "18" },
  { key: "throughput_mbps", label: "Throughput", unit: "Mbps", initial: "50" },
] as const;

export type KpiKey = (typeof KPI_FIELDS)[number]["key"];

export const DEFAULT_KPIS = Object.fromEntries(
  KPI_FIELDS.map((field) => [field.key, field.initial]),
) as Record<KpiKey, string>;
