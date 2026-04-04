import pytest

from nalr.runtime.entropy import QuantumEntropyPool, QuantumEntropyUnavailableError


class BrokenEntropyProvider:
    endpoint = "https://broken-qrng.test/api"

    def fetch_batch(self, *, byte_count: int):
        raise RuntimeError(f"provider unavailable for {byte_count} bytes")


def test_quantum_entropy_pool_hard_blocks_when_entropy_unavailable():
    pool = QuantumEntropyPool(provider=BrokenEntropyProvider(), prefetch_bytes=64)

    with pytest.raises(QuantumEntropyUnavailableError) as excinfo:
        pool.uniform(purpose="round-1-action-sample", node_name="action_sample")

    assert "action_sample" in str(excinfo.value)
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
