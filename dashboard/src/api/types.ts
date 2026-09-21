export interface RegistryEntry {
  active_version: string;
  available_versions: string[];
  canary: string | null;
  canary_percent: number;
}
export type RegistryResponse = Record<string, RegistryEntry>;

export interface VersionStats {
  requests_total: number;
  errors_total: number;
  p50: number | null;
  p95: number | null;
  p99: number | null;
}
export type StatsResponse = Record<string, VersionStats>;

export interface DriftStatus {
  metric_name: string;
  score: number;
  breached: boolean;
  consecutive_breaches: number;
  timestamp: number;
}

export interface Incident {
  id: number;
  model_id: string;
  version: string;
  action: "auto_rollback" | "canary_killed" | "breach_no_fallback";
  reason: string;
  metric_name: string | null;
  metric_value: number | null;
  previous_stable: string | null;
  timestamp: number;
}

export interface DriftHistoryPoint {
  timestamp: number;
  score: number;
  breached: number;
}
export interface LatencyPoint {
  timestamp: number;
  latency_ms: number;
}
export interface Incident {
  id: number;
  model_id: string;
  version: string;
  action: "auto_rollback" | "canary_killed" | "breach_no_fallback";
  reason: string;
  previous_stable: string | null;
  timestamp: number;
}

export interface PercentileBucket {
  timestamp: number; p50: number; p95: number; p99: number; error_rate: number; count: number;
}