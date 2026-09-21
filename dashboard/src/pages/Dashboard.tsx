import { useRegistry } from "@/api/hooks";
import { Header } from "@/components/layout/Header";
import { ModelPanel } from "@/components/registry/ModelPanel";
import { IncidentTimeline } from "@/components/incidents/IncidentTimeline";
import { RegistryTable } from "@/components/registry/RegistryTable";
import { PerformancePanel } from "@/components/performance/PerformancePanel";

export function Dashboard() {
  const { data: registry, isLoading, error } = useRegistry();
  if (isLoading) return <div className="p-8 text-sm text-muted">Loading platform state…</div>;
  if (error) return <div className="p-8 text-sm text-alert">Couldn't reach the serving API.</div>;

  return (
    <div className="min-h-screen bg-canvas">
      <Header />
      <main className="max-w-6xl mx-auto px-8 py-8 space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(registry ?? {}).map(([modelId, entry]) => (
            <ModelPanel key={modelId} modelId={modelId} entry={entry} />
          ))}
        </div>
        <IncidentTimeline />
        <RegistryTable />
        <PerformancePanel />
      </main>
    </div>
  );
}