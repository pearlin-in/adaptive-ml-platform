# tests/benchmark_executor_averages.py
import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image

from serving.registry.registry import ModelRegistry


async def benchmark_single_pass(
    registry, model_name, sample, executor, total_requests=300, concurrency=30
):
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

    rps = total_requests / wall
    p99 = np.percentile(latencies, 99)
    return rps, p99


async def benchmark_averaged(
    registry,
    model_name,
    sample,
    max_workers,
    runs_per_config=5,
    total_requests=300,
    concurrency=30,
):
    executor = ThreadPoolExecutor(max_workers=max_workers)

    # Warm-up run (discarded)
    await benchmark_single_pass(
        registry, model_name, sample, executor, total_requests, concurrency
    )

    rps_list = []
    p99_list = []

    for _ in range(runs_per_config):
        rps, p99 = await benchmark_single_pass(
            registry, model_name, sample, executor, total_requests, concurrency
        )
        rps_list.append(rps)
        p99_list.append(p99)

    executor.shutdown(wait=True)
    return np.mean(rps_list), np.std(rps_list), np.mean(p99_list)


async def run_suite_for_model(registry, model_name, sample, worker_counts, runs=5):
    print(f"\n==========================================================================")
    print(f" EXECUTOR AVERAGES: [{model_name.upper()}] ({runs} Runs + 1 Warm-up per Config)")
    print(f" System Logical Cores: {os.cpu_count()}")
    print(f"==========================================================================")
    print(f" {'max_workers':<12} | {'Mean RPS':<12} | {'Std Dev':<10} | {'Mean P99 (ms)':<14}")
    print(f"--------------------------------------------------------------------------")

    for workers in worker_counts:
        mean_rps, std_rps, mean_p99 = await benchmark_averaged(
            registry, model_name, sample, max_workers=workers, runs_per_config=runs
        )
        print(f" {workers:<12d} | {mean_rps:<12.2f} | ±{std_rps:<9.2f} | {mean_p99:<14.2f}")

    print(f"==========================================================================\n")


async def main():
    torch.set_num_threads(1)  # Single-threaded intra-op execution
    registry = ModelRegistry()

    fraud_sample = {"Time": 10.0, "Amount": 150.0, **{f"V{i}": 0.01 for i in range(1, 29)}}
    sat_sample = Image.fromarray(np.uint8(np.random.rand(64, 64, 3) * 255))
    text_sample = "System architecture performance evaluation and micro-batching benchmark."

    worker_counts = [2, 4, 8, 16, 32, 64, 128]

    # Run for Satellite and AI Text
    await run_suite_for_model(registry, "satellite", sat_sample, worker_counts)
    await run_suite_for_model(registry, "ai_text", text_sample, worker_counts)


if __name__ == "__main__":
    asyncio.run(main())