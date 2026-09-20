# tests/benchmark_all_models.py
import asyncio
import time
import numpy as np
import torch
import torch.nn as nn
from serving.batcher import MicroBatcher


# -------------------------------------------------------------------
# 1. Real/Representative Heavy Model Stubs
# -------------------------------------------------------------------
class MockNeuralVisionModel:
    """Simulates a PyTorch ResNet-style ConvNet forward pass."""
    def __init__(self):
        # 3-layer CNN on 3x64x64 dummy images
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10)
        )
        self.net.eval()

    def predict_batch(self, batch: list[dict]) -> list[dict]:
        # Convert list of single images into a 4D batch tensor: (B, 3, 64, 64)
        tensors = [torch.from_numpy(item["image"]).float() for item in batch]
        batch_tensor = torch.stack(tensors)
        
        with torch.no_grad():
            out = self.net(batch_tensor)
            
        return [{"logits": row.tolist()} for row in out]

    def predict(self, item: dict) -> dict:
        return self.predict_batch([item])[0]


class MockNeuralTextModel:
    """Simulates a PyTorch Transformer/Embedding forward pass."""
    def __init__(self):
        # Linear projection representing embedding layer overhead
        self.encoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.encoder.eval()

    def predict_batch(self, batch: list[dict]) -> list[dict]:
        # Batch tensor: (B, 128)
        tensors = [torch.from_numpy(item["tokens"]).float() for item in batch]
        batch_tensor = torch.stack(tensors)
        
        with torch.no_grad():
            out = self.encoder(batch_tensor)
            
        return [{"embedding": row.tolist()} for row in out]

    def predict(self, item: dict) -> dict:
        return self.predict_batch([item])[0]


class MockTabularTreeModel:
    """Simulates sub-millisecond LightGBM / Decision Tree evaluation."""
    def predict_batch(self, batch: list[dict]) -> list[dict]:
        # Fast numerical operations across array
        return [{"score": float(item["Amount"] * 0.001)} for item in batch]

    def predict(self, item: dict) -> dict:
        return {"score": float(item["Amount"] * 0.001)}


# -------------------------------------------------------------------
# 2. Benchmark Runner
# -------------------------------------------------------------------
def generate_sample_input(model_type: str):
    if model_type == "satellite":
        return {"image": np.random.randn(3, 64, 64).astype(np.float32)}
    elif model_type == "ai_text":
        return {"tokens": np.random.randn(128).astype(np.float32)}
    else:  # fraud
        return {"Time": 1.0, "Amount": 250.0}


async def benchmark_model(model_name: str, model_obj, total_requests: int = 400, concurrency: int = 40):
    semaphore = asyncio.Semaphore(concurrency)

    # --- A. UNBATCHED (Direct ThreadPool execution, zero queueing delay) ---
    unbatched_latencies = []
    loop = asyncio.get_running_loop()

    async def unbatched_worker():
        async with semaphore:
            sample = generate_sample_input(model_name)
            t0 = time.perf_counter()
            # Direct executor call with no batching queue
            await loop.run_in_executor(None, model_obj.predict, sample)
            unbatched_latencies.append((time.perf_counter() - t0) * 1000.0)

    t_start = time.perf_counter()
    await asyncio.gather(*[unbatched_worker() for _ in range(total_requests)])
    unbatched_wall = time.perf_counter() - t_start
    unbatched_rps = total_requests / unbatched_wall

    # --- B. BATCHED (MicroBatcher with 10ms window, max batch 16) ---
    batcher = MicroBatcher(model=model_obj, max_latency_ms=10.0, max_batch_size=16)
    await batcher.start()
    batched_latencies = []

    async def batched_worker():
        async with semaphore:
            sample = generate_sample_input(model_name)
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

    print(f"\n==========================================================")
    print(f" MODEL: [{model_name.upper()}]")
    print(f"==========================================================")
    print(f" Unbatched Throughput : {unbatched_rps:8.2f} req/sec | p99: {p99_unbatched:6.2f} ms")
    print(f" Batched Throughput   : {batched_rps:8.2f} req/sec | p99: {p99_batched:6.2f} ms")
    print(f" Throughput Speedup   : {speedup:8.2f}x")
    print(f"==========================================================")


async def main():
    print("\n🚀 Benchmarking Micro-Batcher across all 3 Modalities...\n")
    
    await benchmark_model("fraud", MockTabularTreeModel())
    await benchmark_model("satellite", MockNeuralVisionModel())
    await benchmark_model("ai_text", MockNeuralTextModel())


if __name__ == "__main__":
    asyncio.run(main())