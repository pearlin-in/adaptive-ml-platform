import { useRegistry } from "@/api/hooks";
import { ModelCard } from "@/components/registry/ModelCard";

export function Dashboard() {
  const { data: registry, isLoading, error } = useRegistry();

  if (isLoading) return <div className="p-8 text-neutral-500">Loading…</div>;
  if (error) return <div className="p-8 text-red-600">Failed to reach the API.</div>;

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <h1 className="text-2xl font-semibold text-neutral-900 mb-6">Model Registry</h1>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {Object.entries(registry ?? {}).map(([modelId, entry]) => (
          <ModelCard key={modelId} modelId={modelId} entry={entry} />
        ))}
      </div>
    </div>
  );
}