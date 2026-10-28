"""
utils/performance.py
────────────────────
FPS tracking and performance diagnostics.
"""

from __future__ import annotations
import time
from collections import deque
from typing import Optional


class FPSTracker:
    """
    Sliding-window FPS counter.
    Measures actual pipeline throughput, not target FPS.

    Uses a fixed-size deque of recent frame timestamps so the reported
    FPS smoothly follows real throughput without abrupt jumps.
    """

    def __init__(self, window_size: int = 30):
        self._timestamps: deque = deque(maxlen=window_size)
        self._window = window_size

    def tick(self):
        """Call once per processed frame."""
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def __str__(self) -> str:
        return f"{self.fps:.1f} FPS"


class StageTimer:
    """
    Context manager for timing individual pipeline stages.

    Usage:
        timer = StageTimer()
        with timer.measure("yolo"):
            run_yolo()
        print(timer.report())
    """

    def __init__(self):
        self._times: dict[str, float] = {}
        self._active: Optional[str] = None
        self._start: float = 0.0

    class _Ctx:
        def __init__(self, parent: "StageTimer", name: str):
            self._p = parent
            self._n = name

        def __enter__(self):
            self._p._active = self._n
            self._p._start = time.perf_counter()

        def __exit__(self, *_):
            elapsed_ms = (time.perf_counter() - self._p._start) * 1000
            self._p._times[self._n] = elapsed_ms
            self._p._active = None

    def measure(self, name: str) -> "_Ctx":
        return self._Ctx(self, name)

    def report(self) -> str:
        parts = [f"{k}={v:.1f}ms" for k, v in self._times.items()]
        return " | ".join(parts)

    def get(self, name: str) -> float:
        return self._times.get(name, 0.0)
