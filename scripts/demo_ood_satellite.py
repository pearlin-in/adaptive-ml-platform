import asyncio, io, random, time
import numpy as np
import httpx
from PIL import Image, ImageOps
from datasets import load_dataset
from scripts.demo_utils import BASE_URL, sustained_send, watch_for_breach_and_rollback

print("Loading a small EuroSAT sample for realistic baseline traffic...")
_eurosat = load_dataset("tanganke/eurosat")["train"]
_sample_indices = random.sample(range(len(_eurosat)), 300)

def baseline_image_bytes() -> bytes:
    img = _eurosat[random.choice(_sample_indices)]["image"].convert("RGB")
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()

def shifted_image_bytes() -> bytes:
    # Simulated different sensor/preprocessing pipeline: inverted color, extreme
    # contrast, injected noise — a documented real domain-shift failure mode.
    img = _eurosat[random.choice(_sample_indices)]["image"].convert("RGB")
    img = ImageOps.autocontrast(ImageOps.invert(img), cutoff=20)
    arr = np.clip(np.array(img).astype(np.int16) + np.random.normal(0, 60, np.array(img).shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()

async def main():
    async with httpx.AsyncClient(timeout=15.0) as client:
        async def send_baseline():
            await client.post(f"{BASE_URL}/predict/satellite",
                               files={"file": ("baseline.png", baseline_image_bytes(), "image/png")})
        async def send_shifted():
            await client.post(f"{BASE_URL}/predict/satellite",
                               files={"file": ("shifted.png", shifted_image_bytes(), "image/png")})

        print("=== Satellite OOD Injection Demo ===")
        print("\nPhase 1: baseline traffic (real EuroSAT images)")
        await sustained_send(send_baseline, duration_s=45, rate_per_s=4, label="satellite-baseline")

        print(f"\nBaseline drift: {await client.get(f'{BASE_URL}/drift/satellite', params={'version': 'v2'})}")

        start_ts = time.time()
        print("\nPhase 2: injecting inverted/high-contrast/noisy images (simulated sensor shift)")
        await asyncio.gather(
            sustained_send(send_shifted, duration_s=90, rate_per_s=5, label="satellite-injection"),
            watch_for_breach_and_rollback(client, "satellite", "v2", start_ts, timeout_s=120),
        )

if __name__ == "__main__":
    asyncio.run(main())