import { useQuery } from "@tanstack/react-query";
import { api } from "./client";

const POLL_INTERVAL_MS = 5000;

export function useRegistry() {
  return useQuery({
    queryKey: ["registry"],
    queryFn: api.getRegistry,
    refetchInterval: POLL_INTERVAL_MS,
  });
}

export function useStats(modelId: string) {
  return useQuery({
    queryKey: ["stats", modelId],
    queryFn: () => api.getStats(modelId),
    refetchInterval: POLL_INTERVAL_MS,
  });
}

export function useDrift(modelId: string, version = "v1") {
  return useQuery({
    queryKey: ["drift", modelId, version],
    queryFn: () => api.getDrift(modelId, version),
    refetchInterval: POLL_INTERVAL_MS,
  });
}

export function useIncidents() {
  return useQuery({
    queryKey: ["incidents"],
    queryFn: () => api.getIncidents(),
    refetchInterval: POLL_INTERVAL_MS,
  });
}