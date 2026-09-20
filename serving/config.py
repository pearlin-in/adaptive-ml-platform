# serving/config.py
from concurrent.futures import ThreadPoolExecutor
import os

# Dedicated worker pools tailored to physical/logical core alignment & variance control
MODEL_EXECUTORS = {
    "fraud": ThreadPoolExecutor(max_workers=8, thread_name_prefix="exec_fraud"),
    "satellite": ThreadPoolExecutor(max_workers=2, thread_name_prefix="exec_satellite"),
    "ai_text": ThreadPoolExecutor(max_workers=16, thread_name_prefix="exec_text"),
}

# Micro-batcher SLAs balancing batch throughput with latency caps
BATCH_CONFIG = {
    "fraud": {
        "max_latency_ms": 10.0,
        "max_batch_size": 32,
    },
    "satellite": {
        "max_latency_ms": 10.0,
        "max_batch_size": 32,
    },
    "ai_text": {
        "max_latency_ms": 10.0,
        "max_batch_size": 8,
    },
}