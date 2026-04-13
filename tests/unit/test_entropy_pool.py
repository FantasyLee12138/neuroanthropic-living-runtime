import pytest

from nalr.runtime.entropy import (
    MacOSSystemEntropyProvider,
    QuantumEntropyPool,
    QuantumEntropyUnavailableError,
)


class BrokenEntropyProvider:
    endpoint = "file:///dev/urandom"

    def fetch_batch(self, *, byte_count: int):
        raise RuntimeError(f"provider unavailable for {byte_count} bytes")


def test_macos_entropy_provider_fetches_bytes_from_os_urandom(monkeypatch):
    provider = MacOSSystemEntropyProvider()
    monkeypatch.setattr("os.urandom", lambda byte_count: bytes(range(byte_count)))

    batch, data = provider.fetch_batch(byte_count=16)

    assert data == bytes(range(16))
    assert batch.source == "macos_os_urandom"
    assert batch.provider_class == "MacOSSystemEntropyProvider"
    assert batch.endpoint == "file:///dev/urandom"
    assert batch.health_state == "ready"
    assert batch.degraded is False


def test_quantum_entropy_pool_hard_blocks_when_entropy_unavailable():
    pool = QuantumEntropyPool(provider=BrokenEntropyProvider(), prefetch_bytes=64)

    with pytest.raises(QuantumEntropyUnavailableError) as excinfo:
        pool.uniform(purpose="round-1-action-sample", node_name="action_sample")

    assert "action_sample" in str(excinfo.value)
    health = pool.health_snapshot()
    assert health["state"] == "blocked"
    assert health["last_failure"]["node_name"] == "action_sample"
    assert health["last_failure"]["failure_class"] == "provider_fetch_failed"


def test_quantum_entropy_pool_still_blocks_when_hard_block_disabled_but_provider_fails():
    pool = QuantumEntropyPool(
        provider=BrokenEntropyProvider(),
        prefetch_bytes=64,
        hard_block_on_unavailable=False,
    )

    with pytest.raises(QuantumEntropyUnavailableError):
        pool.uniform(purpose="round-1-action-sample", node_name="action_sample")

    health = pool.health_snapshot()
    assert health["state"] == "blocked"
    assert health["last_failure"]["node_name"] == "action_sample"
    assert health["last_failure"]["failure_class"] == "provider_fetch_failed"


def test_quantum_entropy_pool_records_node_specific_audit_fields():
    pool = QuantumEntropyPool(prefetch_bytes=64)
    pool.ingest_bytes(bytes(range(64)), source="fixture_qrng", reason="unit test seed")

    value, ref = pool.uniform(purpose="round-2-action-sample", node_name="action_sample")

    assert 0.0 <= value < 1.0
    assert ref.source == "fixture_qrng"
    assert ref.purpose == "round-2-action-sample"
    assert ref.node_name == "action_sample"
    assert ref.provider_class
    assert ref.health_state == "ready"
    assert ref.hard_block_triggered is False


def test_quantum_entropy_pool_uses_macos_provider_by_default(monkeypatch):
    monkeypatch.setattr("os.urandom", lambda byte_count: bytes([7]) * byte_count)
    pool = QuantumEntropyPool(prefetch_bytes=64)

    value, ref = pool.uniform(purpose="round-2-action-sample", node_name="action_sample")

    assert 0.0 <= value < 1.0
    assert ref.source == "macos_os_urandom"
    assert ref.provider_class == "MacOSSystemEntropyProvider"
    assert ref.endpoint == "file:///dev/urandom"


def test_macos_entropy_provider_raises_clear_error_when_os_urandom_fails(monkeypatch):
    provider = MacOSSystemEntropyProvider()

    def broken_urandom(byte_count: int) -> bytes:
        raise OSError(f"os urandom unavailable for {byte_count}")

    monkeypatch.setattr("os.urandom", broken_urandom)

    with pytest.raises(OSError) as excinfo:
        provider.fetch_batch(byte_count=16)

    assert "os urandom unavailable" in str(excinfo.value)
