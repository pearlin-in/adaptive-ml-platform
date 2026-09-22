// src/components/performance/PerformancePanel.tsx
import { useState } from "react";
import { ComposedChart, Line, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from "recharts";
import { usePercentileHistory } from "@/api/hooks";
import type { RegistryResponse } from "@/api/types";

const MODELS = ["fraud", "satellite", "ai_text"] as const;
function formatTime(ts: number) {
  return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function PerformancePanel({ registry }: { registry: RegistryResponse }) {
  const [activeModel, setActiveModel] = useState<(typeof MODELS)[number]>("fraud");
  const activeVersion = registry[activeModel]?.active_version ?? "v1";
  const { data } = usePercentileHistory(activeModel, activeVersion);
  const chartData = (data ?? []).map((b) => ({ ...b, time: formatTime(b.timestamp), error_pct: b.error_rate * 100 }));

  return (
    <div className="bg-surface border border-line rounded-sm">
      <div className="flex items-center justify-between px-5 pt-4">
        <h2 className="text-sm font-medium text-ink">Latency & error rate</h2>
        <div className="flex gap-4">
          {MODELS.map((m) => (
            <button key={m} onClick={() => setActiveModel(m)}
              className={`text-xs pb-2 border-b-2 transition-colors ${
                activeModel === m ? "border-ink text-ink font-medium" : "border-transparent text-muted hover:text-ink"
              }`}>
              {m.replace("_", " ")}
            </button>
          ))}
        </div>
      </div>

      <div className="h-56 px-3 pt-2">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="2 4" stroke="var(--color-line)" vertical={false} />
            <XAxis dataKey="time" tick={{ fontSize: 10, fill: "var(--color-muted)" }} axisLine={{ stroke: "var(--color-line)" }} tickLine={false} />
            <YAxis yAxisId="latency" tick={{ fontSize: 10, fill: "var(--color-muted)" }} axisLine={false} tickLine={false} width={36}
              label={{ value: "ms", angle: -90, position: "insideLeft", fontSize: 10, fill: "var(--color-muted)" }} />
            <YAxis yAxisId="error" orientation="right" tick={{ fontSize: 10, fill: "var(--color-muted)" }} axisLine={false} tickLine={false} width={32}
              label={{ value: "err %", angle: 90, position: "insideRight", fontSize: 10, fill: "var(--color-muted)" }} />
            <Tooltip contentStyle={{ fontSize: 12, borderRadius: 2, border: "1px solid var(--color-line)" }} labelStyle={{ color: "var(--color-muted)" }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar yAxisId="error" dataKey="error_pct" name="error rate %" fill="var(--color-alert-soft)" stroke="var(--color-alert)" barSize={8} />
            <Line yAxisId="latency" type="monotone" dataKey="p50" name="p50" stroke="var(--color-stable)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            <Line yAxisId="latency" type="monotone" dataKey="p95" name="p95" stroke="var(--color-canary)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            <Line yAxisId="latency" type="monotone" dataKey="p99" name="p99" stroke="var(--color-alert)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      {chartData.length === 0 && <div className="px-5 pb-4 text-xs text-muted">No traffic recorded yet for this model.</div>}
    </div>
  );
}