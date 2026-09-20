# tests/benchmark_real_models.py
import asyncio
import time
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from serving.batcher import MicroBatcher
from serving.registry.registry import ModelRegistry


# -------------------------------------------------------------------
# 1. Satellite Model Preprocessing Sub-step Profiler
# -------------------------------------------------------------------
def profile_satellite_preprocessing(batch_size: int = 16):
    print("\n--- [PROFILING] Satellite Model Sub-step Breakdown ---")
    
    # Generate mock PIL images (EuroSAT default resolution: 64x64)
    raw_images = [Image.fromarray(np.uint8(np.random.rand(64, 64, 3) * 255)) for _ in range(batch_size)]
    
    resize = transforms.Resize((64, 64))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Sub-step 1: RGB Conversion
    t0 = time.perf_counter()
    imgs = [img.convert("RGB") for img in raw_images]
    t1 = time.perf_counter()

    # Sub-step 2: Resize
    resized = [resize(im) for im in imgs]
    t2 = time.perf_counter()

    # Sub-step 3: ToTensor (Memory copy + Float conversion)
    tensors = [to_tensor(im) for im in resized]
    t3 = time.perf_counter()

    # Sub-step 4: Vectorized Stacking + Broadcasted Normalization
    batch_tensor = torch.stack(tensors)
    normalized_batch = normalize(batch_tensor)
    t4 = time.perf_counter()

    print(f"  • convert('RGB')  : {(t1 - t0) * 1000:.3f} ms")
    print(f"  • resize(64, 64)  : {(t2 - t1) * 1000:.3f} ms")
    print(f"  • to_tensor()     : {(t3 - t2) * 1000:.3f} ms")
    print(f"  • stack + norm    : {(t4 - t3) * 1000:.3f} ms (Vectorized Call)")
    print(f"  • Total Preprocess: {(t4 - t0) * 1000:.3f} ms\n")


# -------------------------------------------------------------------
# 2. End-to-End Real Model Benchmarker
# -------------------------------------------------------------------
async def benchmark_registered_model(
    registry: ModelRegistry, 
    model_name: str, 
    total_requests: int = 300, 
    concurrency: int = 30
):
    model = registry.get(model_name)
    semaphore = asyncio.Semaphore(concurrency)
    loop = asyncio.get_running_loop()

    # Payload generator using exact input signature expected by each model's predict/predict_batch
    if model_name == "fraud":
        sample = {"Time": 10.0, "Amount": 150.0}
        for i in range(1, 29):
            sample[f"V{i}"] = 0.01
    elif model_name == "satellite":
        # SatelliteModel expects a direct PIL Image instance
        sample = Image.fromarray(np.uint8(np.random.rand(64, 64, 3) * 255))
    else:  # ai_text
        # AITextModel expects a direct text string
        sample = "System architecture performance evaluation and micro-batching benchmark."

    # --- A. UNBATCHED BASELINE (Direct per-item calls) ---
    unbatched_latencies = []

    async def unbatched_worker():
        async with semaphore:
            t0 = time.perf_counter()
            await loop.run_in_executor(None, model.predict, sample)
            unbatched_latencies.append((time.perf_counter() - t0) * 1000.0)

    t_start = time.perf_counter()
    await asyncio.gather(*[unbatched_worker() for _ in range(total_requests)])
    unbatched_wall = time.perf_counter() - t_start
    unbatched_rps = total_requests / unbatched_wall

    # --- B. BATCHED BASELINE (MicroBatcher with 10ms window) ---
    batcher = MicroBatcher(model=model, max_latency_ms=10.0, max_batch_size=16)
    await batcher.start()
    batched_latencies = []

    async def batched_worker():
        async with semaphore:
            t0 = time.perf_counter()
            await batcher.submit(sample)
            batched_latencies.append((time.perf_counter() - t0) * 1000.0)

    t_start = time.perf_counter()
    await asyncio.gather(*[batched_worker() for _ in range(total_requests)])
    batched_wall = time.perf_counter() - t_start
    await batcher.stop()

    batched_rps = total_requests / batched_wall
    speedup = batched_rps / unbatched_rps
    p99_unbatched = np.percentile(unbatched_latencies, 99)
    p99_batched = np.percentile(batched_latencies, 99)

    print(f"==========================================================")
    print(f" REAL MODEL BENCHMARK: [{model_name.upper()}]")
    print(f"==========================================================")
    print(f" Unbatched Throughput : {unbatched_rps:8.2f} req/sec | p99: {p99_unbatched:6.2f} ms")
    print(f" Batched Throughput   : {batched_rps:8.2f} req/sec | p99: {p99_batched:6.2f} ms")
    print(f" Throughput Speedup   : {speedup:8.2f}x")
    print(f"==========================================================\n")


async def main():
    profile_satellite_preprocessing(batch_size=16)
    
    print("🚀 Initializing ModelRegistry & Running Benchmarks...\n")
    registry = ModelRegistry()

    await benchmark_registered_model(registry, "fraud")
    await benchmark_registered_model(registry, "satellite")
    await benchmark_registered_model(registry, "ai_text")


if __name__ == "__main__":
    asyncio.run(main())