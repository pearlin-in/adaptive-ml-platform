"""
scripts/run_full_demo.py

Runs the three OOD drift demos back to back, with a clean server restart and state reset
before each one, then verifies each model produced exactly one auto_rollback.

    python -m scripts.run_full_demo                 # unattended
    python -m scripts.run_full_demo --interactive   # prints screenshot cues and pauses first

Before running:
  - Stop any running uvicorn (this script starts and stops the API itself).
  - Keep the dashboard (npm run dev) open in your browser.
  - Optional, for clean charts: back up and delete serving/predictions.db once.

Outputs: docs/demo_logs/*.txt|log, docs/demo_results.json, docs/demo_results.md
"""
import argparse
import json
import os
import subprocess
import sys
import time

import httpx

BASE = "http://localhost:8000"
MODELS = ["fraud", "satellite", "ai_text"]
SIGNALS = {"fraud": "PSI (max feature)", "satellite": "cosine distance", "ai_text": "cosine distance"}


def api_get(path: str, **params):
    r = httpx.get(f"{BASE}{path}", params=params, timeout=10.0)
    r.raise_for_status()
    return r.json()


def server_up() -> bool:
    try:
        return httpx.get(f"{BASE}/registry", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


def start_server(log_path: str):
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "serving.app:app"],
        stdout=log, stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 180
    while time.time() < deadline:
        if proc.poll() is not None:
            log.close()
            raise RuntimeError(f"Server exited early; see {log_path}")
        if server_up():
            return proc, log
        time.sleep(2)
    proc.terminate()
    log.close()
    raise RuntimeError("Server did not become ready within 180s")


def run_demo(model: str, log_path: str) -> dict:
    """Run the model's demo script, stream its output, and print screenshot cues."""
    marks, cued = {}, set()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", f"scripts.demo_ood_{model}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
    )
    with open(log_path, "w", encoding="utf-8") as log:
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            if "Phase 2" in line and "start" not in marks:
                marks["start"] = time.time()
            if "Baseline drift:" in line and "before" not in cued:
                cued.add("before")
                print(f"\n>>> SCREENSHOT NOW (before): {model} baseline, all models nominal, v2 active\n")
            if "<-- BREACH" in line and "during" not in cued:
                cued.add("during")
                print(f"\n>>> SCREENSHOT NOW (during): {model} panel border + drift timeline crossing threshold\n")
            if "INCIDENT RECORDED" in line and "after" not in cued:
                cued.add("after")
                print(f"\n>>> SCREENSHOT NOW (after): {model} registry shows v1 + incident log entry\n")
    proc.wait()
    return marks


def collect(model: str, start_ts: float, marks: dict, thresholds: dict) -> dict:
    incidents = api_get("/incidents", model_id=model, since=start_ts - 1)
    rollbacks = [i for i in incidents if i["action"] == "auto_rollback"]
    history = api_get(f"/drift/{model}/history", version="v2", limit=200)
    scores = [p["score"] for p in history if p["timestamp"] >= start_ts]
    active_after = api_get("/registry")[model]["active_version"]
    injection_start = marks.get("start", start_ts)
    return {
        "model": model,
        "signal": SIGNALS[model],
        "threshold": thresholds.get(model),
        "peak_score": max(scores) if scores else None,
        "auto_rollbacks": len(rollbacks),
        "other_incidents": len(incidents) - len(rollbacks),
        "reason": rollbacks[-1]["reason"] if rollbacks else None,
        "seconds_injection_to_rollback": (
            round(rollbacks[-1]["timestamp"] - injection_start, 1) if rollbacks else None
        ),
        "active_version_after": active_after,
        "passed": len(rollbacks) == 1 and active_after == "v1",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactive", action="store_true",
                    help="pause before each demo so the dashboard can be arranged")
    args = ap.parse_args()

    if server_up():
        sys.exit("A server is already running on :8000. Stop it first; this script manages the API itself.")

    os.makedirs("docs/demo_logs", exist_ok=True)
    results = []

    for model in MODELS:
        print(f"\n{'=' * 70}\n {model}\n{'=' * 70}")
        subprocess.run([sys.executable, "-m", "scripts.reset_for_demo"], check=True)
        proc, log = start_server(f"docs/demo_logs/server_{model}.log")
        try:
            registry = api_get("/registry")
            not_v2 = [m for m, e in registry.items() if e["active_version"] != "v2"]
            if not_v2:
                raise RuntimeError(f"Server did not pick up the reset; not on v2: {not_v2}")
            try:
                thresholds = api_get("/thresholds")
            except httpx.HTTPError:
                thresholds = {}
            if args.interactive:
                input("Dashboard ready? Press Enter to start the demo... ")
            start_ts = time.time()
            marks = run_demo(model, f"docs/demo_logs/{model}.txt")
            results.append(collect(model, start_ts, marks, thresholds))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()
            time.sleep(3)  # let the port free up

    with open("docs/demo_results.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = [
        "| Model | Signal | Threshold | Peak score | Injection to rollback | Rolled back to | Passed |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        peak = f"{r['peak_score']:.4f}" if r["peak_score"] is not None else "n/a"
        secs = f"{r['seconds_injection_to_rollback']}s" if r["seconds_injection_to_rollback"] is not None else "n/a"
        lines.append(
            f"| {r['model']} | {r['signal']} | {r['threshold']} | {peak} | {secs} | "
            f"{r['active_version_after']} | {'yes' if r['passed'] else 'NO'} |"
        )
    summary = "\n".join(lines)
    with open("docs/demo_results.md", "w") as f:
        f.write(summary + "\n")

    print(f"\n{'=' * 70}\n SUMMARY\n{'=' * 70}\n{summary}\n")
    for r in results:
        if not r["passed"]:
            print(f"FAILED: {r['model']}: {r['auto_rollbacks']} auto_rollback(s), "
                  f"{r['other_incidents']} other incident(s), active={r['active_version_after']}. "
                  f"See docs/demo_logs/{r['model']}.txt")
    sys.exit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()