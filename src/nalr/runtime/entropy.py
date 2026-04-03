from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any
from uuid import uuid4
import time

import httpx

from nalr.runtime.metadata import utc_now_iso
from nalr.schemas.models import QuantumEntropyBatch, QuantumEntropyRef


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class _BufferState:
    batch: QuantumEntropyBatch
    bytes_queue: deque[int]
    cursor: int = 0


class AnuQuantumEntropyProvider:
    def __init__(self, endpoint: str = "https://qrng.anu.edu.au/API/jsonI.php", timeout_s: float = 0.6) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    def fetch_batch(self, *, byte_count: int) -> QuantumEntropyBatch:
        params = {"length": min(max(int(byte_count), 32), 1024), "type": "uint8"}
        fetched_at = utc_now_iso()
        response = httpx.get(self.endpoint, params=params, timeout=self.timeout_s)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(f"qrng provider returned unsuccessful payload: {payload}")
        data = payload.get("data", [])
        if not isinstance(data, list) or not data:
            raise RuntimeError("qrng provider returned empty payload")
        values = bytes(int(item) & 0xFF for item in data)
        return QuantumEntropyBatch(
            source="anu_qrng",
            fetched_at=fetched_at,
            batch_id=f"anu-{uuid4().hex[:12]}",
            total_bytes=len(values),
            available_bytes=len(values),
            degraded=False,
            reason="",
        ), values


class QuantumEntropyPool:
    def __init__(self, provider: Any | None = None, *, prefetch_bytes: int = 256) -> None:
        self.provider = provider or AnuQuantumEntropyProvider()
        self.prefetch_bytes = max(prefetch_bytes, 64)
        self._state: _BufferState | None = None
        self._failure_reason = ""
        self._disable_fetch_until = 0.0

    def ingest_bytes(self, data: bytes, *, source: str = "test_qrng", reason: str = "") -> None:
        batch = QuantumEntropyBatch(
            source=source,
            fetched_at=utc_now_iso(),
            batch_id=f"{source}-{uuid4().hex[:12]}",
            total_bytes=len(data),
            available_bytes=len(data),
            degraded=False,
            reason=reason,
        )
        self._state = _BufferState(batch=batch, bytes_queue=deque(data), cursor=0)

    def _ensure(self, length: int) -> tuple[bytes, QuantumEntropyRef]:
        if self._state is None or len(self._state.bytes_queue) < length:
            if self._disable_fetch_until > time.monotonic():
                return b"", QuantumEntropyRef(
                    source=getattr(self.provider, "endpoint", "quantum_pool"),
                    fetched_at=utc_now_iso(),
                    batch_id="degraded",
                    byte_start=0,
                    byte_length=0,
                    degraded=True,
                    reason=self._failure_reason or "qrng_fetch_cooldown",
                )
            try:
                batch, data = self.provider.fetch_batch(byte_count=max(length, self.prefetch_bytes))
                self._state = _BufferState(batch=batch, bytes_queue=deque(data), cursor=0)
                self._failure_reason = ""
                self._disable_fetch_until = 0.0
            except Exception as exc:
                self._failure_reason = str(exc)
                self._disable_fetch_until = time.monotonic() + 60.0
                return b"", QuantumEntropyRef(
                    source=getattr(self.provider, "endpoint", "quantum_pool"),
                    fetched_at=utc_now_iso(),
                    batch_id="degraded",
                    byte_start=0,
                    byte_length=0,
                    degraded=True,
                    reason=self._failure_reason,
                )

        start = self._state.cursor
        taken = bytes(self._state.bytes_queue.popleft() for _ in range(length))
        self._state.cursor += length
        self._state.batch.available_bytes = len(self._state.bytes_queue)
        return taken, QuantumEntropyRef(
            source=self._state.batch.source,
            fetched_at=self._state.batch.fetched_at,
            batch_id=self._state.batch.batch_id,
            byte_start=start,
            byte_length=length,
            degraded=self._state.batch.degraded,
            reason=self._state.batch.reason,
        )

    def uniform(self, *, purpose: str = "") -> tuple[float, QuantumEntropyRef]:
        raw, ref = self._ensure(8)
        if not raw:
            return 0.5, ref
        integer = int.from_bytes(raw, byteorder="big", signed=False)
        value = integer / float(2**64)
        return min(value, 1.0 - 1e-12), ref

    def uniform_range(self, low: float, high: float, *, purpose: str = "") -> tuple[float, QuantumEntropyRef]:
        base, ref = self.uniform(purpose=purpose)
        return low + (high - low) * base, ref

    def truncated_normal(self, *, sigma: float, low: float, high: float, purpose: str = "") -> tuple[float, QuantumEntropyRef]:
        sigma = max(float(sigma), 1e-6)
        dist = NormalDist(mu=0.0, sigma=sigma)
        low_cdf = dist.cdf(low)
        high_cdf = dist.cdf(high)
        base, ref = self.uniform(purpose=purpose)
        if ref.degraded:
            return _clip(0.0, low, high), ref
        sample_cdf = low_cdf + (high_cdf - low_cdf) * base
        sample = dist.inv_cdf(min(max(sample_cdf, 1e-12), 1.0 - 1e-12))
        return _clip(sample, low, high), ref

    def beta_like(self, *, mu: float, kappa: float, purpose: str = "") -> tuple[float, QuantumEntropyRef]:
        base, ref = self.uniform(purpose=purpose)
        if ref.degraded:
            return _clip(mu, 0.05, 0.95), ref
        alpha = max(mu * kappa, 0.25)
        beta = max((1.0 - mu) * kappa, 0.25)
        sample = (1.0 - (1.0 - base) ** (1.0 / beta)) ** (1.0 / alpha)
        return _clip(sample, 0.05, 0.95), ref

    def weighted_choice(self, distribution: dict[str, float], *, purpose: str = "") -> tuple[str, QuantumEntropyRef]:
        threshold, ref = self.uniform(purpose=purpose)
        cumulative = 0.0
        fallback = ""
        for name, value in sorted(distribution.items()):
            probability = max(float(value), 0.0)
            cumulative += probability
            fallback = name
            if threshold <= cumulative:
                return name, ref
        return fallback, ref
