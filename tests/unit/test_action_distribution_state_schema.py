from dataclasses import fields

from nalr.schemas.models import ActionBookkeepingState


def test_action_bookkeeping_state_only_keeps_base_bookkeeping_fields():
    names = {item.name for item in fields(ActionBookkeepingState)}

    assert {"u_base", "p_base", "ci", "gate", "risk_suppressor", "query_intent", "disclosure_intent"} <= names
    assert "u_shifted" not in names
    assert "p_raw" not in names
    assert "p_mix" not in names
    assert "p_final" not in names
    assert "p_base_stochastic" not in names
    assert "q_noise" not in names
    assert "conflict_mode" not in names
    assert "conflict" not in names
