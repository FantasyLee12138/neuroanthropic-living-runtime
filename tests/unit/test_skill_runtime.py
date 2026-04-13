import json
import shutil
import time
from pathlib import Path

import nalr.skills.executor as skill_executor_module
from nalr.providers.router import MissingModelCredentialError
from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import (
    CircuitBreakerPolicy,
    FallbackRoute,
    ProbabilisticContribution,
    RoundEvent,
    SkillPermissionProfile,
    SkillRuntimeContext,
    SkillSpec,
)
from nalr.skills.executor import SkillExecutor
from nalr.skills.registry import build_skill_registry


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"
VALID_RUNTIME_INPUTS = {
    "event": {"source": "user", "content": "help me plan", "cue": "plan"},
    "state": {"mode": "interactive", "round_count": 0},
    "scenario": {"pfc_base_share": 0.3},
    "context": {"cue": "plan", "recall_strength": 0.2, "closeness": 0.5},
}


def test_skill_executor_applies_fallback_policy():
    executor = SkillExecutor(build_skill_registry())

    result = executor.execute(
        round_id=1,
        skill_name="generate_candidates",
        inputs=VALID_RUNTIME_INPUTS,
        provider=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert result.degraded is True
    assert result.failure_policy_applied == "fallback_to_rules"
    assert result.output


def test_skill_executor_rejects_invalid_output_schema():
    executor = SkillExecutor(build_skill_registry())

    result = executor.execute(
        round_id=1,
        skill_name="generate_candidates",
        inputs=VALID_RUNTIME_INPUTS,
        provider=lambda: {"unexpected": "shape"},
    )

    assert result.degraded is True
    assert result.failure_policy_applied == "output_validation_failed"


def test_skill_executor_keeps_successful_output_even_when_latency_exceeds_budget():
    executor = SkillExecutor(build_skill_registry())

    output, result = executor.run(
        round_id=1,
        skill_name="request_second_sampling",
        inputs={"fail_score": 0.4, "attempts": 0},
        provider=lambda fail_score, attempts: (time.sleep(0.02), {"flag": fail_score > 0.2})[1],
    )

    assert output["flag"] is True
    assert result.degraded is False
    assert result.output["flag"] is True
    assert result.latency_ms >= 20


def test_skill_executor_trips_circuit_breaker_after_repeated_failures():
    executor = SkillExecutor(build_skill_registry())

    for _ in range(3):
        result = executor.execute(
            round_id=1,
            skill_name="generate_candidates",
            inputs=VALID_RUNTIME_INPUTS,
            provider=lambda: {"unexpected": "shape"},
        )
        assert result.degraded is True

    breaker_result = executor.execute(
        round_id=2,
        skill_name="generate_candidates",
        inputs=VALID_RUNTIME_INPUTS,
        provider=lambda: {"action_preferences": {"plan": 0.4}},
    )

    assert breaker_result.degraded is True
    assert breaker_result.failure_policy_applied == "trip_circuit_breaker"


def test_skill_executor_does_not_poison_breaker_when_model_credentials_are_missing(tmp_path):
    executor = SkillExecutor(build_skill_registry(), circuit_breaker_path=tmp_path / "circuit_breakers.json")

    for round_id in range(1, 4):
        output, result = executor.run(
            round_id=round_id,
            skill_name="generate_candidates",
            inputs=VALID_RUNTIME_INPUTS,
            provider=lambda: (_ for _ in ()).throw(MissingModelCredentialError("ARK_API_KEY is required")),
        )
        assert result.degraded is True
        assert result.failure_policy_applied == "fallback_to_rules"
        assert result.policy_rejection_reason == "missing_model_credentials"
        assert result.breaker_state["failure_count"] == 0
        assert result.breaker_state["open_until_round"] is None
        assert output


def test_hash_payload_is_stable_across_dict_order():
    payload_a = {
        "event": {"source": "user", "content": "hello"},
        "context": {"cue": "plan", "score": 0.4},
    }
    payload_b = {
        "context": {"score": 0.4, "cue": "plan"},
        "event": {"content": "hello", "source": "user"},
    }

    assert skill_executor_module._hash_payload(payload_a) == skill_executor_module._hash_payload(payload_b)


def test_hash_payload_distinguishes_long_strings_beyond_prefix():
    prefix = "脑" * 200
    payload_a = {"text": f"{prefix}A"}
    payload_b = {"text": f"{prefix}B"}

    assert skill_executor_module._hash_payload(payload_a) != skill_executor_module._hash_payload(payload_b)


def test_hash_payload_avoids_full_contract_serialization_for_large_runtime_inputs(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    runtime_state = controller.load_runtime_state()
    event = RoundEvent(source="user", content="最近的状态如何？", target="user", cue="状态")

    def _unexpected_serializer(_value):
        raise AssertionError("hash path should not use serialize_contract_value")

    monkeypatch.setattr(skill_executor_module, "serialize_contract_value", _unexpected_serializer)

    digest = skill_executor_module._hash_payload(
        {
            "event": event,
            "state": runtime_state,
            "scenario": {"pfc_base_share": 0.3},
            "context": {"cue": "状态", "recall_strength": 0.1},
        }
    )

    assert len(digest) == 40
    assert all(character in "0123456789abcdef" for character in digest)


def test_tick_records_skill_level_trace_and_files(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Help me remember tea and plan tomorrow.",
            target="user",
            cue="tea",
            valence=0.2,
        ),
        scenario="task",
        mode="interactive",
    )

    assert result.trace.skill_traces
    skills_path = tmp_path / ".alive" / "traces" / "skills" / "skill_traces.jsonl"
    assert skills_path.exists()
    rows = [json.loads(line) for line in skills_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(row["round_id"] == result.round_id for row in rows)


def test_tick_uses_heuristic_pfc_fallback_when_model_provider_fails(tmp_path):
    config_root = tmp_path / "config"
    shutil.copytree(CONFIG_ROOT, config_root)
    controller = RuntimeController(project_root=tmp_path, config_root=config_root)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Help me plan dinner and remember rice.",
            target="user",
            cue="rice",
            valence=0.1,
        ),
        scenario="task",
        mode="interactive",
    )

    generate_candidates_trace = next(
        item for item in result.trace.skill_traces if item["skill_name"] == "generate_candidates"
    )
    pfc_summary = next(item for item in result.trace.proposal_summaries if item["stage"] == "pfc")

    assert generate_candidates_trace["degraded"] is True
    assert pfc_summary["action_preferences"].get("plan", 0.0) > 0.0
    assert pfc_summary["action_preferences"].get("recall", 0.0) > 0.0


def test_skill_executor_coerces_typed_inputs_and_outputs(tmp_path):
    spec = SkillSpec(
        name="typed_demo",
        owner_module="demo",
        input_schema={"event": RoundEvent},
        output_schema=ProbabilisticContribution,
        timeout_ms=20,
        cost_class="H",
        failure_policy="fallback_to_rules",
        trace_tags=["demo"],
        skill_kind="planning",
        output_kind="contribution",
        policy_check=True,
        permission=SkillPermissionProfile(external_io=True),
        fallback_route=FallbackRoute(strategy="fallback_to_rules", target="demo.fallback", cost_class="L"),
        breaker_policy=CircuitBreakerPolicy(failure_threshold=3, cooldown_rounds=5),
    )
    executor = SkillExecutor({"typed_demo": spec}, circuit_breaker_path=tmp_path / "circuit_breakers.json")

    output, result = executor.run(
        round_id=1,
        skill_name="typed_demo",
        inputs={"event": {"source": "user", "content": "help me plan", "cue": "plan"}},
        provider=lambda event: {
            "module_name": "demo",
            "module_type": "executive",
            "level": "action",
            "target_space": "action",
            "confidence": 0.8,
            "raw_signal": {"plan": 0.3},
            "modulated_delta": {"plan": 0.3},
            "trace_reason": event.content,
            "projection_reason": "typed test",
        },
        fallback_provider=lambda event: ProbabilisticContribution(
            module_name="demo",
            module_type="executive",
            level="action",
            target_space="action",
            raw_signal={"respond": 0.1},
            modulated_delta={"respond": 0.1},
            confidence=0.1,
            trace_reason="fallback",
            projection_reason="typed fallback",
        ),
        runtime_context=SkillRuntimeContext(round_id=1, scenario="task", mode="interactive"),
    )

    assert isinstance(output, ProbabilisticContribution)
    assert output.trace_reason == "help me plan"
    assert result.degraded is False
    assert result.output["modulated_delta"]["plan"] == 0.3
    assert {"raw_signal", "modulated_delta", "inhibitory_drive", "failure_taxonomy"}.issubset(result.output)


def test_skill_executor_persists_breaker_and_uses_fallback_during_cooldown(tmp_path):
    spec = SkillSpec(
        name="typed_breaker",
        owner_module="demo",
        input_schema={"event": RoundEvent},
        output_schema=ProbabilisticContribution,
        timeout_ms=20,
        cost_class="H",
        failure_policy="fallback_to_rules",
        trace_tags=["demo"],
        skill_kind="planning",
        output_kind="contribution",
        policy_check=True,
        permission=SkillPermissionProfile(external_io=True),
        fallback_route=FallbackRoute(strategy="fallback_to_rules", target="demo.fallback", cost_class="L"),
        breaker_policy=CircuitBreakerPolicy(failure_threshold=3, cooldown_rounds=5),
    )
    breaker_path = tmp_path / "circuit_breakers.json"
    executor = SkillExecutor({"typed_breaker": spec}, circuit_breaker_path=breaker_path)
    context = SkillRuntimeContext(round_id=1, scenario="task", mode="interactive")

    for round_id in range(1, 4):
        _, result = executor.run(
            round_id=round_id,
            skill_name="typed_breaker",
            inputs={"event": {"source": "user", "content": "help me plan"}},
            provider=lambda event: (_ for _ in ()).throw(RuntimeError("boom")),
            fallback_provider=lambda event: ProbabilisticContribution(
                module_name="demo",
                module_type="executive",
                level="action",
                target_space="action",
                raw_signal={"plan": 0.2},
                modulated_delta={"plan": 0.2},
                confidence=0.2,
                trace_reason="fallback",
                projection_reason="typed fallback",
            ),
            runtime_context=SkillRuntimeContext(round_id=round_id, scenario="task", mode="interactive"),
        )
        assert result.degraded is True

    persisted = json.loads(breaker_path.read_text(encoding="utf-8"))
    assert persisted["typed_breaker"]["open_until_round"] == 8

    called_primary = {"count": 0}
    output, breaker_result = executor.run(
        round_id=4,
        skill_name="typed_breaker",
        inputs={"event": {"source": "user", "content": "help me plan"}},
        provider=lambda event: called_primary.__setitem__("count", called_primary["count"] + 1),
        fallback_provider=lambda event: ProbabilisticContribution(
            module_name="demo",
            module_type="executive",
            level="action",
            target_space="action",
            raw_signal={"plan": 0.25},
            modulated_delta={"plan": 0.25},
            confidence=0.25,
            trace_reason="fallback",
            projection_reason="typed fallback",
        ),
        runtime_context=SkillRuntimeContext(round_id=4, scenario="task", mode="interactive"),
    )

    assert called_primary["count"] == 0
    assert isinstance(output, ProbabilisticContribution)
    assert breaker_result.failure_policy_applied == "trip_circuit_breaker"
    assert breaker_result.fallback_route == "demo.fallback"
    assert breaker_result.breaker_state["open"] is True


def test_skill_executor_resets_legacy_open_breakers_after_guard_fingerprint_changes(tmp_path):
    spec = SkillSpec(
        name="typed_breaker",
        owner_module="demo",
        input_schema={"event": RoundEvent},
        output_schema=ProbabilisticContribution,
        timeout_ms=20,
        cost_class="H",
        failure_policy="fallback_to_rules",
        trace_tags=["demo"],
        skill_kind="planning",
        output_kind="contribution",
        policy_check=True,
        permission=SkillPermissionProfile(external_io=True),
        fallback_route=FallbackRoute(strategy="fallback_to_rules", target="demo.fallback", cost_class="L"),
        breaker_policy=CircuitBreakerPolicy(failure_threshold=3, cooldown_rounds=5),
    )
    breaker_path = tmp_path / "circuit_breakers.json"
    breaker_path.write_text(
        json.dumps(
            {
                "typed_breaker": {
                    "failure_count": 3,
                    "open_until_round": 10,
                    "last_failure_round": 5,
                    "last_failure_reason": "fallback_to_rules",
                    "fallback_route": "demo.fallback",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    executor = SkillExecutor(
        {"typed_breaker": spec},
        circuit_breaker_path=breaker_path,
        environment_fingerprint="guard-v2",
    )

    called_primary = {"count": 0}
    output, result = executor.run(
        round_id=6,
        skill_name="typed_breaker",
        inputs={"event": {"source": "user", "content": "help me plan"}},
        provider=lambda event: called_primary.__setitem__("count", called_primary["count"] + 1) or ProbabilisticContribution(
            module_name="demo",
            module_type="executive",
            level="action",
            target_space="action",
            confidence=0.7,
            raw_signal={"plan": 0.2},
            modulated_delta={"plan": 0.2},
            trace_reason=event.content,
            projection_reason="typed test",
        ),
        fallback_provider=lambda event: ProbabilisticContribution(
            module_name="demo",
            module_type="executive",
            level="action",
            target_space="action",
            raw_signal={"respond": 0.1},
            modulated_delta={"respond": 0.1},
            confidence=0.1,
            trace_reason="fallback",
            projection_reason="typed fallback",
        ),
        runtime_context=SkillRuntimeContext(round_id=6, scenario="task", mode="interactive"),
    )

    persisted = json.loads(breaker_path.read_text(encoding="utf-8"))

    assert called_primary["count"] == 1
    assert isinstance(output, ProbabilisticContribution)
    assert result.degraded is False
    assert persisted["__meta__"]["environment_fingerprint"] == "guard-v2"

    recovered, recovered_result = executor.run(
        round_id=9,
        skill_name="typed_breaker",
        inputs={"event": {"source": "user", "content": "help me plan"}},
        provider=lambda event: ProbabilisticContribution(
            module_name="demo",
            module_type="executive",
            level="action",
            target_space="action",
            raw_signal={"plan": 0.4},
            modulated_delta={"plan": 0.4},
            confidence=0.4,
            trace_reason="primary",
            projection_reason="typed test",
        ),
        fallback_provider=lambda event: ProbabilisticContribution(
            module_name="demo",
            module_type="executive",
            level="action",
            target_space="action",
            raw_signal={"plan": 0.2},
            modulated_delta={"plan": 0.2},
            confidence=0.2,
            trace_reason="fallback",
            projection_reason="typed fallback",
        ),
        runtime_context=SkillRuntimeContext(round_id=9, scenario="task", mode="interactive"),
    )

    assert isinstance(recovered, ProbabilisticContribution)
    assert recovered.modulated_delta["plan"] == 0.4
    assert recovered_result.degraded is False


def test_skill_executor_blocks_policy_denied_external_route():
    spec = SkillSpec(
        name="policy_demo",
        owner_module="demo",
        input_schema={"event": RoundEvent},
        output_schema=ProbabilisticContribution,
        timeout_ms=20,
        cost_class="H",
        failure_policy="fallback_to_rules",
        trace_tags=["demo"],
        skill_kind="planning",
        output_kind="contribution",
        policy_check=True,
        permission=SkillPermissionProfile(external_io=True, social_risk=True),
        fallback_route=FallbackRoute(strategy="fallback_to_rules", target="demo.fallback", cost_class="L"),
        breaker_policy=CircuitBreakerPolicy(failure_threshold=3, cooldown_rounds=5),
    )
    executor = SkillExecutor({"policy_demo": spec})
    called_primary = {"count": 0}

    output, result = executor.run(
        round_id=1,
        skill_name="policy_demo",
        inputs={"event": {"source": "user", "content": "help me plan", "target": "user"}},
        provider=lambda event: called_primary.__setitem__("count", called_primary["count"] + 1),
        fallback_provider=lambda event: ProbabilisticContribution(
            module_name="demo",
            module_type="executive",
            level="action",
            target_space="action",
            raw_signal={"respond": 0.1},
            modulated_delta={"respond": 0.1},
            confidence=0.1,
            trace_reason="fallback",
            projection_reason="typed fallback",
        ),
        runtime_context=SkillRuntimeContext(
            round_id=1,
            scenario="task",
            mode="interactive",
            policy_flags={"deny_external_io": True, "deny_social_inference": True},
        ),
    )

    assert called_primary["count"] == 0
    assert isinstance(output, ProbabilisticContribution)
    assert result.degraded is True
    assert result.policy_rejection_reason == "external_io_denied"


def test_skill_executor_serializes_default_contribution_fallback_without_legacy_aliases(tmp_path):
    spec = SkillSpec(
        name="typed_default",
        owner_module="demo",
        input_schema={"event": RoundEvent},
        output_schema=ProbabilisticContribution,
        timeout_ms=20,
        cost_class="H",
        failure_policy="fallback_to_rules",
        trace_tags=["demo"],
        skill_kind="planning",
        output_kind="contribution",
        policy_check=True,
        permission=SkillPermissionProfile(external_io=True),
        fallback_route=FallbackRoute(strategy="fallback_to_rules", target="demo.fallback", cost_class="L"),
        breaker_policy=CircuitBreakerPolicy(failure_threshold=3, cooldown_rounds=5),
    )
    executor = SkillExecutor({"typed_default": spec}, circuit_breaker_path=tmp_path / "circuit_breakers.json")

    output, result = executor.run(
        round_id=1,
        skill_name="typed_default",
        inputs={"event": {"source": "user", "content": "help me plan"}},
        provider=lambda event: (_ for _ in ()).throw(RuntimeError("boom")),
        runtime_context=SkillRuntimeContext(round_id=1, scenario="task", mode="interactive"),
    )

    assert isinstance(output, ProbabilisticContribution)
    assert output.raw_signal == {"respond": 0.0}
    assert output.modulated_delta == {"respond": 0.0}
    assert output.native_operator == "modulated_delta"
    assert result.output["raw_signal"] == {"respond": 0.0}
    assert result.output["modulated_delta"] == {"respond": 0.0}
    assert result.output["inhibitory_drive"] == {}
    assert result.output["failure_taxonomy"] == []
    assert "raw_signal" in result.output
    assert "modulated_delta" in result.output


def test_skill_executor_serializes_canonical_contribution_without_alias_postcheck(tmp_path):
    spec = SkillSpec(
        name="typed_alias_postcheck",
        owner_module="demo",
        input_schema={"event": RoundEvent},
        output_schema=ProbabilisticContribution,
        timeout_ms=20,
        cost_class="H",
        failure_policy="fallback_to_rules",
        trace_tags=["demo"],
        skill_kind="planning",
        output_kind="contribution",
        policy_check=True,
        permission=SkillPermissionProfile(external_io=True),
        fallback_route=FallbackRoute(strategy="fallback_to_rules", target="demo.fallback", cost_class="L"),
        breaker_policy=CircuitBreakerPolicy(failure_threshold=3, cooldown_rounds=5),
    )
    executor = SkillExecutor({"typed_alias_postcheck": spec}, circuit_breaker_path=tmp_path / "circuit_breakers.json")

    output, result = executor.run(
        round_id=1,
        skill_name="typed_alias_postcheck",
        inputs={"event": {"source": "user", "content": "help me plan"}},
        provider=lambda event: ProbabilisticContribution(
            module_name="demo",
            module_type="executive",
            level="action",
            target_space="action",
            raw_signal={"plan": 0.3},
            modulated_delta={"plan": 0.3},
            inhibitory_drive={"rest": 0.2},
            confidence=0.6,
            trace_reason=event.content,
            projection_reason="typed test",
        ),
        runtime_context=SkillRuntimeContext(round_id=1, scenario="task", mode="interactive"),
    )

    assert isinstance(output, ProbabilisticContribution)
    assert result.degraded is False
    assert result.output["modulated_delta"] == {"plan": 0.3}
    assert result.output["inhibitory_drive"] == {"rest": 0.2}
    assert {"raw_signal", "modulated_delta", "inhibitory_drive", "failure_taxonomy"}.issubset(result.output)


def test_skill_executor_does_not_rewrite_breaker_file_for_steady_success(tmp_path, monkeypatch):
    spec = SkillSpec(
        name="steady_success",
        owner_module="demo",
        input_schema={"event": RoundEvent},
        output_schema={"flag": bool},
        timeout_ms=20,
        cost_class="L",
        failure_policy="fallback_to_rules",
        trace_tags=["demo"],
        skill_kind="planning",
        output_kind="gate",
        policy_check=True,
        permission=SkillPermissionProfile(external_io=False),
        fallback_route=FallbackRoute(strategy="fallback_to_rules", target="demo.fallback", cost_class="L"),
        breaker_policy=CircuitBreakerPolicy(failure_threshold=3, cooldown_rounds=5),
    )
    executor = SkillExecutor({"steady_success": spec}, circuit_breaker_path=tmp_path / "circuit_breakers.json")
    writes: list[str] = []

    original_write_text_atomic = executor._write_text_atomic

    def tracked_write(path, content):
        writes.append(str(path))
        return original_write_text_atomic(path, content)

    monkeypatch.setattr(executor, "_write_text_atomic", tracked_write)

    executor.execute(
        round_id=1,
        skill_name="steady_success",
        inputs={"event": {"source": "user", "content": "ok"}},
        provider=lambda event: {"flag": True},
    )
    first_write_count = len(writes)
    executor.execute(
        round_id=2,
        skill_name="steady_success",
        inputs={"event": {"source": "user", "content": "ok"}},
        provider=lambda event: {"flag": True},
    )

    assert first_write_count >= 1
    assert len(writes) == first_write_count
