"""Progress lines for work that can sit in one call for more than ten minutes.

A stage or subprocess that prints only when it finishes looks idle, and the
host rule is a line at least every ten minutes. This wraps a blocking call
without changing what it returns.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from types import TracebackType
from typing import TextIO
import sys


DEFAULT_INTERVAL_S = 600.0


class Heartbeat:
    """Print ``still running`` every ``interval_s`` until the block exits.

    ``interval_s is None`` or ``<= 0`` disables the background thread. The
    elapsed line on exit still runs so stage timings have a single format.
    """

    def __init__(
        self,
        label: str,
        *,
        interval_s: float | None = DEFAULT_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
        emit: Callable[[str], None] | None = None,
    ) -> None:
        self.label = label
        self.interval_s = interval_s
        self.clock = clock
        self._emit = emit if emit is not None else _stdout
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def __enter__(self) -> Heartbeat:
        self._t0 = self.clock()
        interval = self.interval_s
        if interval is not None and interval > 0:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        elapsed = self.clock() - self._t0
        status = "failed" if exc_type is not None else "done"
        self._emit(f"{self.label} {status} in {elapsed:.1f}s")

    def _loop(self) -> None:
        interval = float(self.interval_s or 0)
        while not self._stop.wait(interval):
            elapsed = self.clock() - self._t0
            self._emit(f"{self.label} still running elapsed={elapsed:.0f}s")


def _stdout(message: str) -> None:
    stream: TextIO = sys.stdout
    print(message, flush=True, file=stream)
