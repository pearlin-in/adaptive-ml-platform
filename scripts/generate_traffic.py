# scripts/generate_traffic.py
"""
Generates synthetic traffic against the running FastAPI server so the
dashboard has real data. Run `uvicorn serving.app:app` first, then this
in a separate terminal. Also a preview of Phase 9's demo-script structure.

ASSUMES:
  POST /predict/fraud     -> JSON matching FraudInput
  POST /predict/satellite -> multipart upload, field "file"
  POST /predict/ai_text   -> JSON {"text": "..."}
Adjust if your actual request models differ.
"""
import asyncio, io, random
import httpx
import numpy as np
from PIL import Image

BASE_URL = "http://localhost:8000"
REQUESTS_PER_MODEL = 300
CONCURRENCY = 10
ERROR_INJECTION_RATE = 0.05  # fraction of requests deliberately malformed


def random_fraud_payload(malformed=False) -> dict:
    payload = {"Time": random.uniform(0, 172792), "Amount": random.uniform(1, 500),
               **{f"V{i}": random.gauss(0, 1) for i in range(1, 29)}}
    if malformed:
        del payload["Amount"]
    return payload

def random_satellite_image(malformed=False) -> bytes:
    if malformed:
        return b"not a real image"
    arr = (np.random.rand(64, 64, 3) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()

SAMPLE_TEXTS = [
    "The quarterly report shows a marked improvement in operational efficiency.",
    "I think we should reconsider the timeline given the recent setbacks.",
    "As an AI language model, I can provide a comprehensive overview of this topic.",
    "Honestly, I'm not sure this approach is going to work out the way we hoped.",
]
def random_text_payload(malformed=False) -> dict:
    return {"text": ""} if malformed else {"text": random.choice(SAMPLE_TEXTS)}


async def hit_fraud(client, sem):
    async with sem:
        try:
            r = await client.post(f"{BASE_URL}/predict/fraud",
                                   json=random_fraud_payload(random.random() < ERROR_INJECTION_RATE))
            print(f"[fraud] {r.status_code}")
        except Exception as e:
            print(f"[fraud] failed: {e}")

async def hit_satellite(client, sem):
    async with sem:
        malformed = random.random() < ERROR_INJECTION_RATE
        files = {"file": ("sample.png", random_satellite_image(malformed), "image/png")}
        try:
            r = await client.post(f"{BASE_URL}/predict/satellite", files=files)
            print(f"[satellite] {r.status_code}")
        except Exception as e:
            print(f"[satellite] failed: {e}")

async def hit_ai_text(client, sem):
    async with sem:
        try:
            r = await client.post(f"{BASE_URL}/predict/ai_text",
                                   json=random_text_payload(random.random() < ERROR_INJECTION_RATE))
            print(f"[ai_text] {r.status_code}")
        except Exception as e:
            print(f"[ai_text] failed: {e}")

async def main():
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = []
        for _ in range(REQUESTS_PER_MODEL):
            tasks += [hit_fraud(client, sem), hit_satellite(client, sem), hit_ai_text(client, sem)]
        random.shuffle(tasks)
        await asyncio.gather(*tasks)
    print("Done — refresh the dashboard.")

if __name__ == "__main__":
    asyncio.run(main())