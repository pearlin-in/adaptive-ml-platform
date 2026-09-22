import asyncio, random, time
import httpx
from scripts.demo_utils import BASE_URL, sustained_send, watch_for_breach_and_rollback

def normal_fraud_payload():
    return {"Time": random.uniform(0, 172792), "Amount": random.uniform(1, 500),
            **{f"V{i}": random.gauss(0, 1) for i in range(1, 29)}}

def shifted_fraud_payload():
    # Mirrors the synthetic shift that hit PSI=4.2467 in the Phase 6 sanity test:
    # much larger amounts, several PCA components pushed well outside training range.
    payload = {"Time": random.uniform(0, 172792), "Amount": random.uniform(5000, 25000)}
    shifted_features = {1, 3, 4, 7, 10, 12, 14, 17}
    for i in range(1, 29):
        payload[f"V{i}"] = random.gauss(0, 1) + (6.0 if i in shifted_features else 0.0)
    return payload

async def main():
    async with httpx.AsyncClient(timeout=10.0) as client:
        async def send_baseline():
            await client.post(f"{BASE_URL}/predict/fraud", json=normal_fraud_payload())
        async def send_shifted():
            await client.post(f"{BASE_URL}/predict/fraud", json=shifted_fraud_payload())

        print("=== Fraud OOD Injection Demo ===")
        print("\nPhase 1: baseline traffic (filling drift window, in-distribution)")
        await sustained_send(send_baseline, duration_s=40, rate_per_s=6, label="fraud-baseline")

        print(f"\nBaseline drift: {await client.get(f'{BASE_URL}/drift/fraud', params={'version': 'v2'})}")

        start_ts = time.time()
        print("\nPhase 2: injecting shifted transactions (simulated emerging fraud pattern)")
        await asyncio.gather(
            sustained_send(send_shifted, duration_s=90, rate_per_s=8, label="fraud-injection"),
            watch_for_breach_and_rollback(client, "fraud", "v2", start_ts, timeout_s=120),
        )

if __name__ == "__main__":
    asyncio.run(main())