from nalr.runtime.contracts import (
    AgencyStatusPayload,
    MeaningStatusPayload,
    PerformanceHotPathPayload,
    SubjectStatusPayload,
    WorkbenchReadModelPayload,
    WorkbenchRoundPayload,
)
from nalr.schemas.models import RoundTrace, RuntimeState


def test_plan16_contracts_define_runtime_and_workbench_payload_shapes():
    assert "history_burden" in RuntimeState.__annotations__
    assert "arbitration_state" in RuntimeState.__annotations__
    assert "event_log_ref" in RoundTrace.__annotations__
    assert "memory_evidence" in RoundTrace.__annotations__
    assert "arbitration_record" in RoundTrace.__annotations__
    assert "history_burden_delta" in RoundTrace.__annotations__
    assert "snapshot_continuity" in RoundTrace.__annotations__

    assert "subject_kernel" in SubjectStatusPayload.__annotations__
    assert "meaning_system" in MeaningStatusPayload.__annotations__
    assert "agency_loop" in AgencyStatusPayload.__annotations__
    assert "latency" in PerformanceHotPathPayload.__annotations__

    assert "bootstrap" in WorkbenchReadModelPayload.__annotations__
    assert "subject" in WorkbenchReadModelPayload.__annotations__
    assert "meaning" in WorkbenchReadModelPayload.__annotations__
    assert "agency" in WorkbenchReadModelPayload.__annotations__
    assert "performance" in WorkbenchReadModelPayload.__annotations__
    assert "state_truth" in WorkbenchReadModelPayload.__annotations__
    assert "memory_evidence" in WorkbenchReadModelPayload.__annotations__
    assert "competition_evidence" in WorkbenchReadModelPayload.__annotations__
    assert "continuity_evidence" in WorkbenchReadModelPayload.__annotations__

    assert "trace" in WorkbenchRoundPayload.__annotations__
    assert "why" in WorkbenchRoundPayload.__annotations__
    assert "whyNot" in WorkbenchRoundPayload.__annotations__
    assert "state_truth" in WorkbenchRoundPayload.__annotations__
    assert "memory_evidence" in WorkbenchRoundPayload.__annotations__
    assert "competition_evidence" in WorkbenchRoundPayload.__annotations__
    assert "continuity_evidence" in WorkbenchRoundPayload.__annotations__
