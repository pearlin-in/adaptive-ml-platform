const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function apiFetch<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    throw new Error(`API error ${res.status} on ${path}`);
  }
  return res.json();
}

export const api = {
  getRegistry: () => apiFetch<import("./types").RegistryResponse>("/registry"),
  getStats: (modelId: string) =>
    apiFetch<import("./types").StatsResponse>(`/stats/${modelId}`),
  getDrift: (modelId: string, version = "v1") =>
    apiFetch<import("./types").DriftStatus | { status: "insufficient_data" }>(
      `/drift/${modelId}?version=${version}`
    ),
  getDriftHistory: (modelId: string, version = "v1", limit = 100) =>
    apiFetch<import("./types").DriftHistoryPoint[]>(
      `/drift/${modelId}/history?version=${version}&limit=${limit}`
    ),
  getLatencyTimeseries: (modelId: string, version = "v1", limit = 60) =>
    apiFetch<import("./types").LatencyPoint[]>(
      `/stats/${modelId}/timeseries?version=${version}&limit=${limit}`
    ),
  getIncidents: (limit = 50) =>
    apiFetch<import("./types").Incident[]>(`/incidents?limit=${limit}`),
};