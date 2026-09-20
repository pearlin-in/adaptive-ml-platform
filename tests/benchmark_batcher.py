# tests/benchmark_batcher.py
import asyncio
import time
import numpy as np
from serving.batcher import MicroBatcher


class DummyModel:
    def predict_batch(self, batch: list[dict]) -> list[dict]:
        # Simulate matrix vectorization: 10ms total overhead for entire batch
        time.sleep(0.010)
        return [{"score": 0.05} for _ in batch]

    def predict(self, item: dict) -> dict:
        # Single item prediction overhead
        time.sleep(0.010)
        return {"score": 0.05}


async def run_throughput_test(use_batching: bool, total_requests: int = 500, concurrency: int = 50):
    dummy_model = DummyModel()

    if use_batching:
        batcher = MicroBatcher(
            model=dummy_model,
            max_latency_ms=10.0,
            max_batch_size=16,
        )
        await batcher.start()

    latencies = []
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(req_id: int):
        async with semaphore:
            start_time = time.perf_counter()
            item = {"Time": 0.0, "Amount": 100.0, "id": req_id}

            if use_batching:
                await batcher.submit(item)
            else:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, dummy_model.predict, item)

            elapsed = (time.perf_counter() - start_time) * 1000.0
            latencies.append(elapsed)

    test_start = time.perf_counter()
    tasks = [worker(i) for i in range(total_requests)]
    await asyncio.gather(*tasks)
    total_time = time.perf_counter() - test_start

    if use_batching:
        await batcher.stop()

    rps = total_requests / total_time
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    mode_str = "BATCHING ENABLED " if use_batching else "BATCHING DISABLED"
    print(f"\n================= {mode_str} =================")
    print(f"Total Requests  : {total_requests}")
    print(f"Concurrency     : {concurrency}")
    print(f"Total Wall Time : {total_time:.3f} s")
    print(f"Throughput      : {rps:.2f} req/sec")
    print(f"p50 Latency     : {p50:.2f} ms")
    print(f"p95 Latency     : {p95:.2f} ms")
    print(f"p99 Latency     : {p99:.2f} ms")
    print("==========================================================")
    return {"rps": rps, "p50": p50, "p95": p95, "p99": p99}


async def main():
    print("\n🚀 Starting Micro-Batcher Performance Benchmark...\n")

    unbatched_stats = await run_throughput_test(use_batching=False)
    batched_stats = await run_throughput_test(use_batching=True)

    speedup = batched_stats["rps"] / unbatched_stats["rps"]
    p99_reduction = ((unbatched_stats["p99"] - batched_stats["p99"]) / unbatched_stats["p99"]) * 100

    print(f"\n📊 SUMMARY RESULTS:")
    print(f"• Throughput Improvement : {speedup:.2f}x higher req/sec")
    print(f"• p99 Latency Reduction  : {p99_reduction:.1f}% faster tail latency\n")


if __name__ == "__main__":
    asyncio.run(main())