import { LineChart, Line, ResponsiveContainer } from "recharts";
import { useStats, useDrift, useLatencyTimeseries } from "@/api/hooks";
import type { RegistryEntry } from "@/api/types";

const STATUS = {
  stable: { border: "border-line", bg: "bg-stable-soft", text: "text-stable", label: "nominal" },
  canary: { border: "border-canary", bg: "bg-canary-soft", text: "text-canary", label: "canary active" },
  alert: { border: "border-alert", bg: "bg-alert-soft", text: "text-alert", label: "drift breach" },
} as const;

function useModelStatus(entry: RegistryEntry, modelId: string) {
  const { data: drift } = useDrift(modelId, entry.active_version);
  if (drift && !("status" in drift) && drift.breached) return "alert" as const;
  if (entry.canary) return "canary" as const;
  return "stable" as const;
}

function Stat({ label, value, unit }: { label: string; value?: number | null; unit: string }) {
  return (
    <div>
      <div className="font-mono text-lg text-ink tabular-nums">
        {value != null ? value.toFixed(unit === "ms" ? 1 : 0) : "–"}
        {unit && <span className="text-xs text-muted ml-0.5">{unit}</span>}
      </div>
      <div className="text-xs text-muted">{label}</div>
    </div>
  );
}

export function ModelPanel({ modelId, entry }: { modelId: string; entry: RegistryEntry }) {
  const status = useModelStatus(entry, modelId);
  const styles = STATUS[status];
  const { data: stats } = useStats(modelId);
  const { data: latency } = useLatencyTimeseries(modelId, entry.active_version);
  const active = stats?.[entry.active_version];

  return (
    <div className={`bg-surface border ${styles.border} rounded-sm overflow-hidden`}>
      <div className={`flex items-center justify-between px-4 py-2 ${styles.bg}`}>
        <span className="font-medium text-ink capitalize">{modelId.replace("_", " ")}</span>
        <span className={`text-xs font-medium ${styles.text}`}>{styles.label}</span>
      </div>

      <div className="p-4">
        <div className="flex items-baseline justify-between mb-1">
          <span className="text-xs text-muted">version</span>
          <span className="font-mono text-xs text-ink">{entry.active_version}</span>
        </div>
        {entry.canary && (
          <div className="flex items-baseline justify-between mb-2">
            <span className="text-xs text-muted">canary</span>
            <span className="font-mono text-xs text-canary">
              {entry.canary} — {entry.canary_percent}%
            </span>
          </div>
        )}

        <div className="grid grid-cols-3 gap-2 my-4">
          <Stat label="p50" value={active?.p50} unit="ms" />
          <Stat label="p99" value={active?.p99} unit="ms" />
          <Stat label="reqs" value={active?.requests_total} unit="" />
        </div>

        <div className="h-10 -mx-1">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={latency ?? []}>
              <Line type="monotone" dataKey="latency_ms" stroke="var(--color-ink)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="text-[10px] text-muted mt-1">latency, last {latency?.length ?? 0} requests</div>
      </div>
    </div>
  );
}