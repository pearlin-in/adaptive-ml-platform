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