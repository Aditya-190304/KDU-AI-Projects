"""Timing helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def capture_elapsed_ms() -> Iterator[list[float]]:
    started = time.perf_counter()
    bucket = [0.0]
    try:
        yield bucket
    finally:
        bucket[0] = round((time.perf_counter() - started) * 1000, 2)
