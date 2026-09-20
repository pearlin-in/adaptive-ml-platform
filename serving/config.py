# serving/config.py
from concurrent.futures import ThreadPoolExecutor
import os

EXECUTORS = {
    "fraud": ThreadPoolExecutor(max_workers=8),
    "satellite": ThreadPoolExecutor(max_workers=4),
    "ai_text": ThreadPoolExecutor(max_workers=8),
}

BATCH_CONFIG = {
    "fraud":     {"max_latency_ms": 10,  "max_batch_size": 32},
    "satellite": {"max_latency_ms": 10,  "max_batch_size": 32},
    "ai_text":   {"max_latency_ms": 10,  "max_batch_size": 8},   # latency-aware pick, not the raw "best throughput" config
}