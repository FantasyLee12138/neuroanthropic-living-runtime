from __future__ import annotations

from typing import Any, TypedDict

from nalr.schemas.models import (
    ActionBookkeepingState,
    ActionEvidenceSignal,
    DisclosureIntentState,
    IdentityContext,
    MotivationPoolState,
    ProbabilisticContribution,
    QueryIntentState,
    RenderedExpression,
    RoundEvent,
    RoundTrace,
    RuntimeState,
    SkillRuntimeContext,
)


class RoundContext(TypedDict):
    event: RoundEvent
    state: RuntimeState
    prior_state: RuntimeState
    latest_round_recorded_at: str | None
    requested_mode: str
    endogenous_turn: bool
    mode_cfg: dict[str, Any]
    scenario_cfg: dict[str, Any]
    thresholds: dict[str, Any]
    prior_closeness: float
    appraisal: dict[str, Any]
    recorded_at: str
    recorded_date: str
    resource_telemetry: dict[str, Any]
    cue: str | None
    memory_write_gate: dict[str, Any]
    context: dict[str, Any]
    relation_state: dict[str, Any]
    shaping_events: list[dict[str, Any]]
    dream_payload: dict[str, Any]
    rename_event: dict[str, Any] | None
    identity_evidence: dict[str, Any]
    gate_decisions: list[dict[str, Any]]
    skill_traces: list[dict[str, Any]]
    model_call_traces: list[dict[str, Any]]
    parallel_traces: list[dict[str, Any]]
    previous_focus: str | None
    round_seed: int
    reasoning_state: RuntimeState
    runtime_context: SkillRuntimeContext
    slow_variables: dict[str, float]


class CollectedContributions(TypedDict):
    query_state: QueryIntentState
    disclosure_state: DisclosureIntentState
    runtime_inputs: dict[str, Any]
    direct_action_contributions: dict[str, ProbabilisticContribution]
    action_signals: list[ActionEvidenceSignal]
    action_bookkeeping: ActionBookkeepingState
    identity_context: IdentityContext
    online_long_run_projection: dict[str, Any]
    motivation_pool_state: MotivationPoolState
    monologue_stream_trace: dict[str, Any]


class ArbitrationResult(TypedDict):
    trace: RoundTrace
    sampled_action: Any
    rendered_expression: RenderedExpression
    control_ledger: dict[str, Any]


class RunQuery(TypedDict, total=False):
    run_id: str
    status: str
    current_step: dict[str, Any]
    stop_reason: dict[str, Any]
    tools: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    trace_ref: str | None
    round_id: int | None


class ObserverSnapshot(TypedDict, total=False):
    state: dict[str, Any]
    autonomy: dict[str, Any]
    action_field: dict[str, Any]
    timeline: dict[str, Any]
    why_current: dict[str, Any]
    why_not: dict[str, Any] | None
    probability_field: dict[str, Any]
    counterfactual_preview: dict[str, Any] | None
    recent_rounds: list[dict[str, Any]]
    source_links: list[dict[str, Any]]


class SubjectStatusPayload(TypedDict):
    subject_kernel: dict[str, Any]
    subject_core_integrity: bool


class MeaningStatusPayload(TypedDict):
    meaning_system: dict[str, Any]
    purpose_memory: list[dict[str, Any]]
    relationship_commitments: list[dict[str, Any]]


class AgencyStatusPayload(TypedDict):
    agency_loop: dict[str, Any]
    proactive_backlog: list[dict[str, Any]]
    initiative: dict[str, Any]
    monologue: dict[str, Any]
    scheduled_tasks: list[dict[str, Any]]


class PerformanceHotPathPayload(TypedDict):
    runtime_metrics: dict[str, Any]
    latency: dict[str, Any]


class WorkbenchReadModelPayload(TypedDict, total=False):
    bootstrap: dict[str, Any]
    summary: dict[str, Any]
    console: ObserverSnapshot
    settings: dict[str, Any] | None
    initiativeStatus: dict[str, Any] | None
    memoryTop: list[dict[str, Any]]
    dreamOverview: dict[str, Any] | None
    monologueShow: dict[str, Any] | None
    sessionState: dict[str, Any] | None
    chatSessionState: dict[str, Any] | None
    subject: SubjectStatusPayload
    meaning: MeaningStatusPayload
    agency: AgencyStatusPayload
    performance: PerformanceHotPathPayload
    inner_space: dict[str, Any]
    state_truth: dict[str, Any]
    memory_evidence: dict[str, Any] | None
    competition_evidence: dict[str, Any] | None
    continuity_evidence: dict[str, Any] | None


class WorkbenchRoundPayload(TypedDict, total=False):
    trace: dict[str, Any]
    initiativeWhy: dict[str, Any]
    thought: dict[str, Any]
    why: dict[str, Any]
    contributions: dict[str, Any]
    probability: dict[str, Any]
    whyNot: dict[str, Any]
    replay: dict[str, Any]
    state_truth: dict[str, Any]
    memory_evidence: dict[str, Any]
    competition_evidence: dict[str, Any]
    continuity_evidence: dict[str, Any]


class EndogenousTickPayload(TypedDict):
    round_id: int | None
    micro_intent: dict[str, Any]
    cause_type: str
    boundary_action: str
    trigger: dict[str, Any]
    selected_mode: str | None
    suppressed: bool
    suppression_reason: str
    trace_ref: str | None
    initiative: dict[str, Any]
