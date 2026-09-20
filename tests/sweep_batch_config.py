# tests/sweep_batch_config.py
import asyncio
import time
import numpy as np
import torch
from PIL import Image

from serving.batcher import MicroBatcher
from serving.registry.registry import ModelRegistry

async def benchmark_batch_params(model, sample, max_batch_size, max_latency_ms, total_requests=300, concurrency=30):
    semaphore = asyncio.Semaphore(concurrency)
    batcher = MicroBatcher(model=model, max_latency_ms=max_latency_ms, max_batch_size=max_batch_size)
    await batcher.start()
    latencies = []

    async def worker():
        async with semaphore:
            t0 = time.perf_counter()
            await batcher.submit(sample)
            latencies.append((time.perf_counter() - t0) * 1000)

    t_start = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(total_requests)])
    wall = time.perf_counter() - t_start
    await batcher.stop()

    rps = total_requests / wall
    p99 = np.percentile(latencies, 99)
    return rps, p99

async def run_sweep_for_model(model_name, sample, registry):
    model = registry.get(model_name)
    batch_sizes = [4, 8, 16, 32]
    latency_windows = [10.0, 25.0, 50.0, 100.0, 150.0]

    print(f"\n=========================================================================")
    print(f" GRID SEARCH BATCH CONFIG SWEEP: [{model_name.upper()}]")
    print(f"=========================================================================")
    print(f" {'Batch Size':<12} | {'Window (ms)':<12} | {'Throughput (req/s)':<20} | {'P99 (ms)':<10}")
    print(f"-------------------------------------------------------------------------")

    best_rps = 0.0
    best_cfg = None

    for bs in batch_sizes:
        for win in latency_windows:
            rps, p99 = await benchmark_batch_params(model, sample, bs, win)
            print(f" {bs:<12d} | {win:<12.1f} | {rps:<20.2f} | {p99:<10.2f}")
            if rps > best_rps:
                best_rps = rps
                best_cfg = (bs, win, p99)

    print(f"-------------------------------------------------------------------------")
    print(f" 🏆 BEST CONFIG FOR {model_name.upper()}: max_batch_size={best_cfg[0]}, max_latency_ms={best_cfg[1]} "
          f"(Throughput: {best_rps:.2f} req/s | P99: {best_cfg[2]:.2f} ms)")
    print(f"=========================================================================\n")

async def main():
    torch.set_num_threads(1)
    registry = ModelRegistry()

    fraud_sample = {"Time": 10.0, "Amount": 150.0, **{f"V{i}": 0.01 for i in range(1, 29)}}
    sat_sample = Image.fromarray(np.uint8(np.random.rand(64, 64, 3) * 255))
    text_sample = "System architecture performance evaluation and micro-batching benchmark."

    await run_sweep_for_model("fraud", fraud_sample, registry)
    await run_sweep_for_model("satellite", sat_sample, registry)
    await run_sweep_for_model("ai_text", text_sample, registry)

if __name__ == "__main__":
    asyncio.run(main())