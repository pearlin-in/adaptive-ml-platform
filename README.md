# adaptive-ml-platform
Multi-modal ML inference server with micro-batching, consistent hash traffic splitting and automated drift mitigation.
things i got to do:
- batching on vs off throughput numbers
- canary traffic split demo & champion/challenger comparison endpoint  comparing accuracy/latency between versions
- write up on drift/monitor.py code had the missing _recompute_all() call in record_sample (so till now drift scoring was dead in production)
- implement the circuit breaker fallback response 
- before/after demo
- deploy/doc