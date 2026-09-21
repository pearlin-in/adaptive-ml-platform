import { useRegistry, useStats } from "@/api/hooks";
import type { RegistryEntry } from "@/api/types";

export function RegistryTable() {
  const { data: registry } = useRegistry();
  return (
    <div className="bg-surface border border-line rounded-sm overflow-hidden">
      <div className="px-5 py-3 border-b border-line">
        <h2 className="text-sm font-medium text-ink">Registry</h2>
      </div>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-muted border-b border-line">
            <th className="font-normal px-5 py-2">model</th>
            <th className="font-normal px-5 py-2">active</th>
            <th className="font-normal px-5 py-2">canary</th>
            <th className="font-normal px-5 py-2">versions</th>
            <th className="font-normal px-5 py-2 text-right">requests</th>
            <th className="font-normal px-5 py-2 text-right">errors</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(registry ?? {}).map(([modelId, entry]) => (
            <RegistryRow key={modelId} modelId={modelId} entry={entry} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RegistryRow({ modelId, entry }: { modelId: string; entry: RegistryEntry }) {
  const { data: stats } = useStats(modelId);
  const active = stats?.[entry.active_version];
  return (
    <tr className="border-b border-line last:border-b-0">
      <td className="px-5 py-2.5 text-ink capitalize">{modelId.replace("_", " ")}</td>
      <td className="px-5 py-2.5 font-mono text-ink">{entry.active_version}</td>
      <td className="px-5 py-2.5 font-mono text-canary">
        {entry.canary ? `${entry.canary} (${entry.canary_percent}%)` : "—"}
      </td>
      <td className="px-5 py-2.5 font-mono text-muted">{entry.available_versions.join(", ")}</td>
      <td className="px-5 py-2.5 font-mono text-ink text-right tabular-nums">{active?.requests_total ?? 0}</td>
      <td className="px-5 py-2.5 font-mono text-right tabular-nums">
        <span className={active?.errors_total ? "text-alert" : "text-muted"}>{active?.errors_total ?? 0}</span>
      </td>
    </tr>
  );
}