import { useEffect, useState } from "react";

export function Header() {
  const [secondsAgo, setSecondsAgo] = useState(0);
  useEffect(() => {
    const start = Date.now();
    const id = setInterval(() => setSecondsAgo(Math.floor((Date.now() - start) / 1000)), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <header className="flex items-center justify-between px-8 py-5 border-b border-line">
      <div>
        <h1 className="text-base font-semibold text-ink">Adaptive ML Platform</h1>
        <p className="text-xs text-muted">Serving, drift detection, and automated recovery</p>
      </div>
      <div className="flex items-center gap-2 text-xs text-muted">
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-stable opacity-75" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-stable" />
        </span>
        refreshed {secondsAgo}s ago
      </div>
    </header>
  );
}