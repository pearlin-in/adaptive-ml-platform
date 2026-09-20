# tests/benchmark_executor_averages.py
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

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

    # Warm-up run (discarded to avoid cold-cache / setup noise)
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

    # Return mean throughput, std dev, and mean P99 latency
    return np.mean(rps_list), np.std(rps_list), np.mean(p99_list)


async def main():
    registry = ModelRegistry()
    fraud_sample = {
        "Time": 10.0,
        "Amount": 150.0,
        **{f"V{i}": 0.01 for i in range(1, 29)},
    }

    worker_counts = [2, 4, 8, 16, 32, 64, 128]
    runs = 5

    print("==========================================================================")
    print(
        f" FRAUD EXECUTOR BENCHMARK (Averaging {runs} Runs + 1 Warm-up per Config)"
    )
    print("==========================================================================")
    print(
        f" {'max_workers':<12} | {'Mean RPS':<12} | {'Std Dev':<10} | {'Mean P99 (ms)':<14}"
    )
    print("--------------------------------------------------------------------------")

    for workers in worker_counts:
        mean_rps, std_rps, mean_p99 = await benchmark_averaged(
            registry, "fraud", fraud_sample, max_workers=workers, runs_per_config=runs
        )
        print(
            f" {workers:<12d} | {mean_rps:<12.2f} | ±{std_rps:<9.2f} | {mean_p99:<14.2f}"
        )

    print("==========================================================================")


if __name__ == "__main__":
    asyncio.run(main())