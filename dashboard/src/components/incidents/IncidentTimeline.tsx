import { useState } from "react";
import { LineChart, Line, XAxis, YAxis, ReferenceLine, ReferenceDot, Tooltip, ResponsiveContainer } from "recharts";
import { useDriftHistory, useIncidents } from "@/api/hooks";
import type { RegistryResponse } from "@/api/types";

const MODELS = ["fraud", "satellite", "ai_text"] as const;
const THRESHOLDS: Record<string, number> = { fraud: 0.25, satellite: 0.15, ai_text: 0.15 };

function formatTime(ts: number) {
  return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function IncidentTimeline({ registry }: { registry: RegistryResponse }) {
  const [activeModel, setActiveModel] = useState<(typeof MODELS)[number]>("fraud");
  const activeVersion = registry[activeModel]?.active_version ?? "v1";
  const { data: history } = useDriftHistory(activeModel, activeVersion);
  const { data: incidents } = useIncidents();

  const modelIncidents = (incidents ?? []).filter((i) => i.model_id === activeModel);
  const chartData = (history ?? []).map((p) => ({ ...p, time: formatTime(p.timestamp) }));
  const breachPoints = chartData.filter((p) => p.breached);

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
                activeModel === m ? "border-ink text-ink font-medium" : "border-transparent text-muted hover:text-ink"
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
            <XAxis dataKey="time" tick={{ fontSize: 10, fill: "var(--color-muted)" }} axisLine={{ stroke: "var(--color-line)" }} tickLine={false} />
            <YAxis tick={{ fontSize: 10, fill: "var(--color-muted)" }} axisLine={false} tickLine={false} width={32} />
            <Tooltip contentStyle={{ fontSize: 12, borderRadius: 2, border: "1px solid var(--color-line)" }} labelStyle={{ color: "var(--color-muted)" }} />
            <ReferenceLine y={THRESHOLDS[activeModel]} stroke="var(--color-alert)" strokeDasharray="3 3"
              label={{ value: "threshold", position: "insideTopRight", fontSize: 10, fill: "var(--color-alert)" }} />
            <Line type="monotone" dataKey="score" stroke="var(--color-ink)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            {breachPoints.map((p, i) => (
              <ReferenceDot key={i} x={p.time} y={p.score} r={3} fill="var(--color-alert)" stroke="none" />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="border-t border-line mt-3">
        {modelIncidents.length === 0 ? (
          <div className="px-5 py-6 text-xs text-muted">No interventions recorded for this model.</div>
        ) : (
          modelIncidents.slice(0, 6).map((incident) => (
            <div key={incident.id} className="flex items-center justify-between px-5 py-2.5 border-b border-line last:border-b-0">
              <div className="flex items-center gap-3">
                <span className={`w-1.5 h-1.5 rounded-full ${incident.action === "auto_rollback" ? "bg-alert" : "bg-canary"}`} />
                <span className="text-xs text-ink">
                  {incident.action === "auto_rollback"
                    ? `Rolled back to ${incident.previous_stable}`
                    : incident.action === "canary_killed"
                    ? `Canary ${incident.version} pulled`
                    : "Breach with no fallback available"}
                </span>
              </div>
              <span className="text-xs font-mono text-muted">{formatTime(incident.timestamp)}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}