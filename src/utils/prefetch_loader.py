"""Background batch prefetch for DALES training loops."""

import queue
import threading


class PrefetchLoader:
    """Keep ``capacity`` batches ready in a queue (CPU→GPU overlap)."""

    def __init__(self, dataset, split, level, batch_size, device, capacity=2):
        self._dataset = dataset
        self._split = split
        self._level = level
        self._batch_size = batch_size
        self._device = device
        self._q = queue.Queue(maxsize=capacity)
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)

    def _worker(self):
        while not self._stop_evt.is_set():
            try:
                batch = self._dataset.sample_batch(
                    self._split, self._level, self._batch_size, self._device
                )
                self._q.put(batch)
            except Exception as e:
                self._q.put(e)

    def start(self):
        self._thread.start()
        return self

    def next(self):
        item = self._q.get()
        if isinstance(item, Exception):
            raise item
        return item

    def stop(self):
        self._stop_evt.set()
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
