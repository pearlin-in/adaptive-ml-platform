import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class MicroBatcher:
    """
    Collects individual predict() requests into a queue and flushes them as one
    predict_batch() call — either when max_batch_size is reached, or when
    max_latency_ms has elapsed since the oldest pending item arrived, whichever
    comes first.
    """

    def __init__(self, model, max_latency_ms: float = 30, max_batch_size: int = 16):
        self.model = model
        self.max_latency_ms = max_latency_ms
        self.max_batch_size = max_batch_size
        self.queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._stopped = False

    async def start(self):
        self._stopped = False
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self):
        self._stopped = True
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def submit(self, raw_input):
        """Called from an endpoint handler. Awaits until this item's result is ready."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await self.queue.put((raw_input, future))
        return await future

    async def _worker_loop(self):
        while not self._stopped:
            try:
                item = await self.queue.get()
            except asyncio.CancelledError:
                break

            batch = [item]
            deadline = time.perf_counter() + (self.max_latency_ms / 1000)

            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    next_item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    batch.append(next_item)
                except asyncio.TimeoutError:
                    break

            await self._process_batch(batch)

    async def _process_batch(self, batch: list[tuple]):
        inputs = [item for item, _ in batch]
        futures = [fut for _, fut in batch]
        loop = asyncio.get_event_loop()

        try:
            # predict_batch is blocking (numpy/torch/sklearn) — never call it directly
            # on the event loop, run it in the default thread-pool executor.
            results = await loop.run_in_executor(None, self.model.predict_batch, inputs)

            if len(results) != len(futures):
                raise RuntimeError(
                    f"predict_batch returned {len(results)} results for {len(futures)} inputs"
                )
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)

        except Exception as batch_exc:
            logger.warning(
                "Batch of %d failed (%s) — falling back to per-item predict() to isolate the bad input",
                len(batch), batch_exc,
            )
            # Fault isolation: one bad input shouldn't take down everyone it happened
            # to be batched with, so fall back to single-item predict() calls.
            for raw_input, future in zip(inputs, futures):
                if future.done():
                    continue
                try:
                    result = await loop.run_in_executor(None, self.model.predict, raw_input)
                    future.set_result(result)
                except Exception as item_exc:
                    future.set_exception(item_exc)