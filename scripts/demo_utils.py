import asyncio, time
import httpx

BASE_URL = "http://localhost:8000"

async def sustained_send(send_one, duration_s: float, rate_per_s: float = 5.0, label: str = ""):
    """Fire send_one() repeatedly at ~rate_per_s for duration_s, fire-and-forget,
    so polling can run concurrently without waiting on each response."""
    interval = 1.0 / rate_per_s
    end_time = time.monotonic() + duration_s
    tasks, sent = [], 0

    async def fire():
        try:
            await send_one()
        except Exception as e:
            print(f"  [{label}] request error (expected sometimes for OOD inputs): {e}")

    while time.monotonic() < end_time:
        tasks.append(asyncio.create_task(fire()))
        sent += 1
        await asyncio.sleep(interval)
    await asyncio.gather(*tasks, return_exceptions=True)
    print(f"  [{label}] sent {sent} requests over {duration_s:.0f}s")
    return sent

async def get_drift(client, model_id, version="v2"):
    try:
        r = await client.get(f"{BASE_URL}/drift/{model_id}", params={"version": version}, timeout=5.0)
        return r.json()
    except Exception:
        return None  # Catch network blips or temporary uvicorn disconnects

async def get_incidents_since(client, model_id, since_ts):
    try:
        r = await client.get(f"{BASE_URL}/incidents", params={"model_id": model_id, "since": since_ts}, timeout=5.0)
        return r.json()
    except Exception:
        return []

async def watch_for_breach_and_rollback(client, model_id, version, start_ts, timeout_s=150, poll_every_s=5):
    print(f"\n--- Watching {model_id} for drift breach (timeout {timeout_s}s) ---")
    elapsed = 0
    breached_seen = False

    while elapsed < timeout_s:
        drift = await get_drift(client, model_id, version)

        if not drift:
            await asyncio.sleep(poll_every_s)
            elapsed += poll_every_s
            continue

        if drift.get("status") == "insufficient_data":
            print(f"  [{elapsed:>3}s] insufficient_data (window still filling)")
        elif "score" in drift:
            is_breached = drift.get("breached", False)
            consecutive = drift.get("consecutive_breaches", 0)
            marker = "  <-- BREACH" if is_breached else ""
            
            print(f"  [{elapsed:>3}s] {drift['metric_name']}={drift['score']:.4f} "
                  f"consecutive_breaches={consecutive}{marker}")
            
            breached_seen = breached_seen or is_breached

            # Check if backend auto-registered an incident
            incidents = await get_incidents_since(client, model_id, start_ts)
            if incidents:
                print("\n  INCIDENT RECORDED:")
                for inc in incidents:
                    print(f"    action={inc['action']} reason=\"{inc['reason']}\" "
                          f"previous_stable={inc['previous_stable']}")
                return True

            # If breach observed but no auto-incident, trigger manually
            if is_breached:
                 print(f"  [{elapsed}s] breach detected (consecutive={consecutive}) — "
                       f"waiting for backend to reach {3} consecutive breaches...")

        await asyncio.sleep(poll_every_s)
        elapsed += poll_every_s

    diag = "drift breached but no incident fired — check IncidentResponder wiring" if breached_seen \
        else "drift score never crossed threshold — consider a stronger injection"
    print(f"  No incident within {timeout_s}s ({diag})")
    return False