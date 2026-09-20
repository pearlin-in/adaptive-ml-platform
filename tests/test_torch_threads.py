# tests/test_torch_threads.py
import asyncio
import time
import torch
import numpy as np
from PIL import Image

from serving.registry.registry import ModelRegistry

async def benchmark_unbatched(registry, model_name, sample, concurrency=30, total_requests=300):
    model = registry.get(model_name)
    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(concurrency)
    latencies = []

    async def worker():
        async with semaphore:
            t0 = time.perf_counter()
            await loop.run_in_executor(None, model.predict, sample)
            latencies.append((time.perf_counter() - t0) * 1000)

    t_start = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(total_requests)])
    wall = time.perf_counter() - t_start
    return total_requests / wall, np.percentile(latencies, 99)

async def main():
    registry = ModelRegistry()
    sat_sample = Image.fromarray(np.uint8(np.random.rand(64, 64, 3) * 255))
    text_sample = "System architecture performance evaluation and micro-batching benchmark."

    print("==========================================================")
    print(" 1. TESTING SATELLITE (CNN)")
    print("==========================================================")
    for threads in [1, 2, 4, torch.get_num_threads()]:
        torch.set_num_threads(threads)
        rps, p99 = await benchmark_unbatched(registry, "satellite", sat_sample)
        print(f" torch_threads={threads:2d} | Throughput={rps:8.2f} req/s | p99={p99:7.2f} ms")

    print("\n==========================================================")
    print(" 2. TESTING AI_TEXT (DISTILBERT)")
    print("==========================================================")
    for threads in [1, 2, 4, torch.get_num_threads()]:
        torch.set_num_threads(threads)
        rps, p99 = await benchmark_unbatched(registry, "ai_text", text_sample)
        print(f" torch_threads={threads:2d} | Throughput={rps:8.2f} req/s | p99={p99:7.2f} ms")

if __name__ == "__main__":
    asyncio.run(main())