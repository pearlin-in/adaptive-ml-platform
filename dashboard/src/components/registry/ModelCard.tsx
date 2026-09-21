import { useStats, useDrift } from "@/api/hooks";
import type { RegistryEntry } from "@/api/types";

function DriftBadge({ modelId, version }: { modelId: string; version: string }) {
  const { data } = useDrift(modelId, version);
  if (!data || "status" in data) {
    return <span className="text-xs text-neutral-400">collecting data…</span>;
  }
  return (
    <span
      className={`text-xs font-medium px-2 py-0.5 rounded-full ${
        data.breached
          ? "bg-red-100 text-red-700"
          : "bg-emerald-100 text-emerald-700"
      }`}
    >
      {data.metric_name}: {data.score.toFixed(3)}
    </span>
  );
}

export function ModelCard({ modelId, entry }: { modelId: string; entry: RegistryEntry }) {
  const { data: stats } = useStats(modelId);
  const activeStats = stats?.[entry.active_version];

  return (
    <div className="rounded-xl border border-neutral-200 bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-neutral-900 capitalize">
          {modelId.replace("_", " ")}
        </h3>
        <DriftBadge modelId={modelId} version={entry.active_version} />
      </div>

      <div className="text-sm text-neutral-500 mb-4">
        active: <span className="font-mono text-neutral-800">{entry.active_version}</span>
        {entry.canary && (
          <>
            {" "}· canary:{" "}
            <span className="font-mono text-amber-700">
              {entry.canary} ({entry.canary_percent}%)
            </span>
          </>
        )}
      </div>

      <div className="grid grid-cols-3 gap-3 text-center">
        <div>
          <div className="text-lg font-semibold text-neutral-900">
            {activeStats?.p50?.toFixed(1) ?? "–"}
          </div>
          <div className="text-xs text-neutral-400">p50 ms</div>
        </div>
        <div>
          <div className="text-lg font-semibold text-neutral-900">
            {activeStats?.p99?.toFixed(1) ?? "–"}
          </div>
          <div className="text-xs text-neutral-400">p99 ms</div>
        </div>
        <div>
          <div className="text-lg font-semibold text-neutral-900">
            {activeStats?.requests_total ?? "–"}
          </div>
          <div className="text-xs text-neutral-400">requests</div>
        </div>
      </div>
    </div>
  );
}