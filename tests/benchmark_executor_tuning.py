# tests/benchmark_executor_tuning.py
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image

from serving.registry.registry import ModelRegistry

async def benchmark_with_executor(registry, model_name, sample, executor, total_requests=300, concurrency=30):
    model = registry.get(model_name)
    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(concurrency)
    latencies = []

    async def worker():
        async with semaphore:
            t0 = time.perf_counter()
            await loop.run_in_executor(executor, model.predict, sample)
            latencies.append((time.perf_counter() - t0) * 1000)

    t_start = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(total_requests)])
    wall = time.perf_counter() - t_start
    return total_requests / wall, np.percentile(latencies, 99)

async def run_executor_sweep(model_name, sample, registry):
    print(f"\n==========================================================")
    print(f" EXECUTOR TUNING SWEEP: [{model_name.upper()}]")
    print(f"==========================================================")
    for max_workers in [2, 4, 8, 16, 32, 64, 128]:
        executor = ThreadPoolExecutor(max_workers=max_workers)
        rps, p99 = await benchmark_with_executor(registry, model_name, sample, executor)
        print(f" max_workers={max_workers:4d} | throughput={rps:8.2f} req/s | p99={p99:7.2f} ms")
        executor.shutdown(wait=True)

async def main():
    # Set single-threaded intra-op PyTorch to avoid thread oversubscription during unbatched runs
    torch.set_num_threads(1)
    
    registry = ModelRegistry()

    fraud_sample = {"Time": 10.0, "Amount": 150.0, **{f"V{i}": 0.01 for i in range(1, 29)}}
    sat_sample = Image.fromarray(np.uint8(np.random.rand(64, 64, 3) * 255))
    text_sample = "System architecture performance evaluation and micro-batching benchmark."

    await run_executor_sweep("fraud", fraud_sample, registry)
    await run_executor_sweep("satellite", sat_sample, registry)
    await run_executor_sweep("ai_text", text_sample, registry)

if __name__ == "__main__":
    asyncio.run(main())