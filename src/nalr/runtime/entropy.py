from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import blake2b
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


class QuantumEntropyUnavailableError(RuntimeError):
    def __init__(self, *, node_name: str, purpose: str, failure_class: str, detail: str) -> None:
        super().__init__(f"entropy unavailable for required probabilistic node '{node_name}' ({purpose}): {detail}")
        self.node_name = node_name
        self.purpose = purpose
        self.failure_class = failure_class
        self.detail = detail


class AnuQuantumEntropyProvider:
    def __init__(
        self,
        endpoint: str = "https://qrng.anu.edu.au/API/jsonI.php",
        timeout_s: float = 0.6,
        min_batch_bytes: int = 32,
        max_batch_bytes: int = 1024,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.min_batch_bytes = max(int(min_batch_bytes), 1)
        self.max_batch_bytes = max(int(max_batch_bytes), self.min_batch_bytes)

    def fetch_batch(self, *, byte_count: int) -> tuple[QuantumEntropyBatch, bytes]:
        params = {"length": min(max(int(byte_count), self.min_batch_bytes), self.max_batch_bytes), "type": "uint8"}
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
            provider_class=self.__class__.__name__,
            endpoint=self.endpoint,
            health_state="ready",
            failure_class="",
            degraded=False,
            reason="",
        ), values


class QuantumEntropyPool:
    def __init__(self, provider: Any | None = None, *, prefetch_bytes: int = 256, hard_block_on_unavailable: bool = True) -> None:
        self.provider = provider or AnuQuantumEntropyProvider()
        self.prefetch_bytes = max(prefetch_bytes, 64)
        self.hard_block_on_unavailable = hard_block_on_unavailable
        self._state: _BufferState | None = None
        self._failure_reason = ""
        self._disable_fetch_until = 0.0
        self._health_state = "cold"
        self._last_failure: dict[str, Any] = {}

    def ingest_bytes(self, data: bytes, *, source: str = "test_qrng", reason: str = "") -> None:
        batch = QuantumEntropyBatch(
            source=source,
            fetched_at=utc_now_iso(),
            batch_id=f"{source}-{uuid4().hex[:12]}",
            total_bytes=len(data),
            available_bytes=len(data),
            provider_class="InjectedEntropyBatch",
            endpoint="memory://ingest",
            health_state="ready",
            failure_class="",
            degraded=False,
            reason=reason,
        )
        self._state = _BufferState(batch=batch, bytes_queue=deque(data), cursor=0)
        self._health_state = "ready"
        self._last_failure = {}

    def _record_failure(self, *, purpose: str, node_name: str, failure_class: str, reason: str) -> None:
        self._health_state = "blocked"
        self._last_failure = {
            "purpose": purpose,
            "node_name": node_name,
            "failure_class": failure_class,
            "reason": reason,
            "recorded_at": utc_now_iso(),
        }

    def _build_deterministic_fallback_batch(self, *, byte_count: int, purpose: str, node_name: str, failure_class: str, reason: str) -> tuple[QuantumEntropyBatch, bytes]:
        total_bytes = max(int(byte_count), self.prefetch_bytes, 64)
        seed = "|".join(
            [
                purpose,
                node_name,
                failure_class,
                reason,
                getattr(self.provider, "endpoint", ""),
                self.provider.__class__.__name__,
            ]
        ).encode("utf-8")
        values = bytearray()
        counter = 0
        while len(values) < total_bytes:
            values.extend(blake2b(seed + counter.to_bytes(4, byteorder="big"), digest_size=32).digest())
            counter += 1
        fetched_at = utc_now_iso()
        batch_id = blake2b(seed, digest_size=6).hexdigest()
        data = bytes(values[:total_bytes])
        return QuantumEntropyBatch(
            source="deterministic_fallback",
            fetched_at=fetched_at,
            batch_id=f"fallback-{batch_id}",
            total_bytes=len(data),
            available_bytes=len(data),
            provider_class=self.provider.__class__.__name__,
            endpoint=getattr(self.provider, "endpoint", ""),
            health_state="degraded",
            failure_class=failure_class,
            degraded=True,
            reason=reason,
        ), data

    def _degrade_or_raise(self, *, length: int, purpose: str, node_name: str, failure_class: str, detail: str) -> None:
        self._record_failure(
            purpose=purpose,
            node_name=node_name,
            failure_class=failure_class,
            reason=detail,
        )
        if self.hard_block_on_unavailable:
            raise QuantumEntropyUnavailableError(
                node_name=node_name,
                purpose=purpose,
                failure_class=failure_class,
                detail=detail,
            )
        batch, data = self._build_deterministic_fallback_batch(
            byte_count=length,
            purpose=purpose,
            node_name=node_name,
            failure_class=failure_class,
            reason=detail,
        )
        self._state = _BufferState(batch=batch, bytes_queue=deque(data), cursor=0)
        self._health_state = "degraded"

    def _ensure(self, length: int, *, purpose: str, node_name: str) -> tuple[bytes, QuantumEntropyRef]:
        if self._state is None or len(self._state.bytes_queue) < length:
            if self._disable_fetch_until > time.monotonic():
                failure_reason = self._failure_reason or "qrng_fetch_cooldown"
                self._degrade_or_raise(
                    length=length,
                    purpose=purpose,
                    node_name=node_name,
                    failure_class="fetch_cooldown",
                    detail=failure_reason,
                )
            try:
                batch, data = self.provider.fetch_batch(byte_count=max(length, self.prefetch_bytes))
                self._state = _BufferState(batch=batch, bytes_queue=deque(data), cursor=0)
                self._failure_reason = ""
                self._disable_fetch_until = 0.0
                self._health_state = "ready"
                self._last_failure = {}
            except Exception as exc:
                self._failure_reason = str(exc)
                self._disable_fetch_until = time.monotonic() + 60.0
                try:
                    self._degrade_or_raise(
                        length=length,
                        purpose=purpose,
                        node_name=node_name,
                        failure_class="provider_fetch_failed",
                        detail=self._failure_reason,
                    )
                except QuantumEntropyUnavailableError as unavailable:
                    raise unavailable from exc

        start = self._state.cursor
        taken = bytes(self._state.bytes_queue.popleft() for _ in range(length))
        self._state.cursor += length
        self._state.batch.available_bytes = len(self._state.bytes_queue)
        self._health_state = "degraded" if self._state.batch.degraded else "ready"
        return taken, QuantumEntropyRef(
            source=self._state.batch.source,
            fetched_at=self._state.batch.fetched_at,
            batch_id=self._state.batch.batch_id,
            byte_start=start,
            byte_length=length,
            purpose=purpose,
            node_name=node_name,
            provider_class=self._state.batch.provider_class,
            endpoint=self._state.batch.endpoint,
            health_state=self._state.batch.health_state,
            hard_block_triggered=False,
            failure_class=self._state.batch.failure_class,
            degraded=self._state.batch.degraded,
            reason=self._state.batch.reason,
        )

    def uniform(self, *, purpose: str = "", node_name: str = "uniform") -> tuple[float, QuantumEntropyRef]:
        raw, ref = self._ensure(8, purpose=purpose, node_name=node_name)
        integer = int.from_bytes(raw, byteorder="big", signed=False)
        value = integer / float(2**64)
        return min(value, 1.0 - 1e-12), ref

    def uniform_range(self, low: float, high: float, *, purpose: str = "", node_name: str = "uniform_range") -> tuple[float, QuantumEntropyRef]:
        base, ref = self.uniform(purpose=purpose, node_name=node_name)
        return low + (high - low) * base, ref

    def truncated_normal(self, *, sigma: float, low: float, high: float, purpose: str = "", node_name: str = "truncated_normal") -> tuple[float, QuantumEntropyRef]:
        sigma = max(float(sigma), 1e-6)
        dist = NormalDist(mu=0.0, sigma=sigma)
        low_cdf = dist.cdf(low)
        high_cdf = dist.cdf(high)
        base, ref = self.uniform(purpose=purpose, node_name=node_name)
        sample_cdf = low_cdf + (high_cdf - low_cdf) * base
        sample = dist.inv_cdf(min(max(sample_cdf, 1e-12), 1.0 - 1e-12))
        return _clip(sample, low, high), ref

    def beta_like(self, *, mu: float, kappa: float, purpose: str = "", node_name: str = "beta_like") -> tuple[float, QuantumEntropyRef]:
        base, ref = self.uniform(purpose=purpose, node_name=node_name)
        alpha = max(mu * kappa, 0.25)
        beta = max((1.0 - mu) * kappa, 0.25)
        sample = (1.0 - (1.0 - base) ** (1.0 / beta)) ** (1.0 / alpha)
        return _clip(sample, 0.05, 0.95), ref

    def weighted_choice(self, distribution: dict[str, float], *, purpose: str = "", node_name: str = "weighted_choice") -> tuple[str, QuantumEntropyRef]:
        threshold, ref = self.uniform(purpose=purpose, node_name=node_name)
        cumulative = 0.0
        fallback = ""
        for name, value in sorted(distribution.items()):
            probability = max(float(value), 0.0)
            cumulative += probability
            fallback = name
            if threshold <= cumulative:
                return name, ref
        return fallback, ref

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "state": self._health_state,
            "provider_class": self.provider.__class__.__name__,
            "endpoint": getattr(self.provider, "endpoint", ""),
            "available_bytes": len(self._state.bytes_queue) if self._state is not None else 0,
            "last_failure": dict(self._last_failure),
        }
