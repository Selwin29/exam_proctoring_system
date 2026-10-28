"""
utils/temporal.py
─────────────────
Frame-counter–based temporal state machines.

Design philosophy:
  Rather than binary flag flips that cause jitter, each behavioral signal
  drives a counter that increments on positive evidence and decays otherwise.
  A "confirmed" event fires only when the counter crosses a trigger threshold.
  This mirrors how human observers would reason: one glance is noise,
  sustained behavior is evidence.

  Counter dynamics resemble a leaky integrator:
    counter[t+1] = clamp(counter[t] + increment, 0, max)   if condition true
    counter[t+1] = clamp(counter[t] - decay,    0, max)    if condition false
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class FrameCounter:
    """
    Single leaky-integrator counter for one behavioral signal.

    Attributes:
        name          : Human-readable label for debug overlay.
        trigger       : Counter value that marks a confirmed event.
        increment     : Amount added per positive frame.
        decay         : Amount subtracted per negative frame.
        max_value     : Hard cap to prevent runaway accumulation.
        value         : Current counter value (float for smooth decay).
        triggered     : True while counter >= trigger threshold.
        trigger_count : Total number of separate trigger events since reset.
    """
    name: str
    trigger: int
    increment: int = 2
    decay: int = 1
    max_value: int = 60

    value: float = field(default=0.0, init=False)
    triggered: bool = field(default=False, init=False)
    trigger_count: int = field(default=0, init=False)
    _was_triggered: bool = field(default=False, init=False)   # edge-detect

    def update(self, condition: bool) -> bool:
        """
        Advance the counter by one frame.

        Args:
            condition: Whether the behavioral signal is active this frame.

        Returns:
            True on the *rising edge* of a trigger event (new detection).
        """
        if condition:
            self.value = min(self.value + self.increment, self.max_value)
        else:
            self.value = max(self.value - self.decay, 0.0)

        self.triggered = self.value >= self.trigger

        # Rising-edge detection — fires once per trigger event, not every frame
        rising_edge = self.triggered and not self._was_triggered
        if rising_edge:
            self.trigger_count += 1

        self._was_triggered = self.triggered
        return rising_edge

    @property
    def fill_ratio(self) -> float:
        """Normalized [0, 1] fill for overlay bar rendering."""
        return min(self.value / self.max_value, 1.0)

    def reset(self):
        self.value = 0.0
        self.triggered = False
        self._was_triggered = False

    def __repr__(self) -> str:
        return (f"FrameCounter({self.name!r}, val={self.value:.1f}, "
                f"triggered={self.triggered}, events={self.trigger_count})")


class TemporalStateManager:
    """
    Manages a named collection of FrameCounter instances.

    Usage:
        tsm = TemporalStateManager()
        tsm.register("look_away", trigger=20, increment=2, decay=1)
        new_event = tsm.update("look_away", condition=True)
        state = tsm.get_state()   # dict for scoring / overlay
    """

    def __init__(self):
        self._counters: Dict[str, FrameCounter] = {}

    def register(
        self,
        name: str,
        trigger: int,
        increment: int = 2,
        decay: int = 1,
        max_value: int = 60,
    ) -> FrameCounter:
        counter = FrameCounter(
            name=name,
            trigger=trigger,
            increment=increment,
            decay=decay,
            max_value=max_value,
        )
        self._counters[name] = counter
        return counter

    def update(self, name: str, condition: bool) -> bool:
        """Update a named counter. Returns True on rising edge."""
        if name not in self._counters:
            raise KeyError(f"Counter '{name}' not registered.")
        return self._counters[name].update(condition)

    def is_triggered(self, name: str) -> bool:
        return self._counters[name].triggered

    def get_value(self, name: str) -> float:
        return self._counters[name].value

    def get_counter(self, name: str) -> FrameCounter:
        return self._counters[name]

    def get_state(self) -> Dict[str, dict]:
        """Snapshot of all counters — used for scoring and overlay."""
        return {
            name: {
                "value": c.value,
                "triggered": c.triggered,
                "fill": c.fill_ratio,
                "events": c.trigger_count,
            }
            for name, c in self._counters.items()
        }

    def reset_all(self):
        for c in self._counters.values():
            c.reset()

    def __repr__(self) -> str:
        lines = ["TemporalStateManager:"]
        for name, c in self._counters.items():
            lines.append(f"  {c}")
        return "\n".join(lines)
