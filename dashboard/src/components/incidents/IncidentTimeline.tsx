import { useState } from "react";
import { useQueries } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ReferenceLine,
  ReferenceDot,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { api } from "@/api/client";
import { useIncidents, useThresholds } from "@/api/hooks";
import type { RegistryResponse } from "@/api/types";

const MODELS = ["fraud", "satellite", "ai_text"] as const;
const POLL_MS = 5000;

function formatTime(ts: number) {
  return new Date(ts * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function IncidentTimeline({ registry }: { registry: RegistryResponse }) {
  const [activeModel, setActiveModel] = useState<(typeof MODELS)[number]>("fraud");
  const versions = registry[activeModel]?.available_versions ?? [];
  const { data: thresholds } = useThresholds();
  const { data: incidents } = useIncidents();

  // One query per version, so a rolled-back version's breach stays visible
  // after traffic has moved back to the previous version.
  const histories = useQueries({
    queries: versions.map((v) => ({
      queryKey: ["drift-history", activeModel, v],
      queryFn: () => api.getDriftHistory(activeModel, v, 150),
      refetchInterval: POLL_MS,
    })),
  });

  const rows = new Map<number, Record<string, number>>();
  histories.forEach((h, i) => {
    (h.data ?? []).forEach((p) => {
      const row = rows.get(p.timestamp) ?? { timestamp: p.timestamp };
      row[`score_${versions[i]}`] = p.score;
      rows.set(p.timestamp, row);
    });
  });
  const chartData = [...rows.values()].sort((a, b) => a.timestamp - b.timestamp);

  const breachDots = histories.flatMap((h, i) =>
    (h.data ?? [])
      .filter((p) => p.breached)
      .map((p) => ({ ...p, key: `${versions[i]}-${p.timestamp}` }))
  );

  const threshold = thresholds?.[activeModel];
  const modelIncidents = (incidents ?? []).filter((i) => i.model_id === activeModel);

  return (
    <div className="bg-surface border border-line rounded-sm">
      <div className="flex items-center justify-between px-5 pt-4">
        <h2 className="text-sm font-medium text-ink">Drift & incident timeline</h2>
        <div className="flex gap-4">
          {MODELS.map((m) => (
            <button
              key={m}
              onClick={() => setActiveModel(m)}
              className={`text-xs pb-2 border-b-2 transition-colors ${
                activeModel === m
                  ? "border-ink text-ink font-medium"
                  : "border-transparent text-muted hover:text-ink"
              }`}
            >
              {m.replace("_", " ")}
            </button>
          ))}
        </div>
      </div>

      <div className="h-64 px-3 pt-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
            <XAxis
              dataKey="timestamp"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={formatTime}
              tick={{ fontSize: 10, fill: "var(--color-muted)" }}
              axisLine={{ stroke: "var(--color-line)" }}
              tickLine={false}
            />
            <YAxis
              domain={[0, (dataMax: number) => Math.max(dataMax, threshold ?? 0) * 1.15]}
              tick={{ fontSize: 10, fill: "var(--color-muted)" }}
              axisLine={false}
              tickLine={false}
              width={36}
              tickFormatter={(v: number) => v.toFixed(2)}
            />
            <Tooltip
              labelFormatter={(ts) => formatTime(Number(ts))}
              contentStyle={{ fontSize: 12, borderRadius: 2, border: "1px solid var(--color-line)" }}
              labelStyle={{ color: "var(--color-muted)" }}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {threshold != null && (
              <ReferenceLine
                y={threshold}
                stroke="var(--color-alert)"
                strokeDasharray="3 3"
                label={{
                  value: `threshold ${threshold}`,
                  position: "insideTopRight",
                  fontSize: 10,
                  fill: "var(--color-alert)",
                }}
              />
            )}
            {versions.map((v, i) => (
              <Line
                key={v}
                type="monotone"
                dataKey={`score_${v}`}
                name={`${v} drift score`}
                connectNulls
                stroke={i === versions.length - 1 ? "var(--color-ink)" : "var(--color-muted)"}
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
              />
            ))}
            {breachDots.map((p) => (
              <ReferenceDot key={p.key} x={p.timestamp} y={p.score} r={3} fill="var(--color-alert)" stroke="none" />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="border-t border-line mt-3">
        {modelIncidents.length === 0 ? (
          <div className="px-5 py-6 text-xs text-muted">No interventions recorded for this model.</div>
        ) : (
          modelIncidents.slice(0, 6).map((incident) => (
            <div
              key={incident.id}
              className="flex items-center justify-between px-5 py-2.5 border-b border-line last:border-b-0"
            >
              <div className="flex items-center gap-3">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    incident.action === "auto_rollback" ? "bg-alert" : "bg-canary"
                  }`}
                />
                <span className="text-xs text-ink">
                  {incident.action === "auto_rollback"
                    ? `Rolled back ${incident.version} to ${incident.previous_stable}`
                    : incident.action === "canary_killed"
                    ? `Canary ${incident.version} pulled`
                    : "Breach with no fallback available"}
                </span>
                <span className="text-xs text-muted">{incident.reason}</span>
              </div>
              <span className="text-xs font-mono text-muted">{formatTime(incident.timestamp)}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}