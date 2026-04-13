from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4


def to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return to_dict(asdict(value))
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    if isinstance(value, tuple):
        return [to_dict(item) for item in value]
    if isinstance(value, set):
        return [to_dict(item) for item in sorted(value)]
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


def _clip_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _looks_like_alias(value: str) -> bool:
    normalized = " ".join((value or "").strip().split())
    if not normalized:
        return False
    if len(normalized) > 20:
        return False
    if any(char in normalized for char in {"\x1b", "\n", "\r", "\t"}):
        return False
    if normalized.startswith("/"):
        return False
    if any(token in normalized for token in {"？", "?", "！", "!", "。"}):
        return False
    return True


def sanitize_identity_aliases(values: list[Any]) -> list[str]:
    cleaned: list[str] = []
    for raw in values:
        if not isinstance(raw, str):
            continue
        normalized = " ".join(raw.strip().split())
        if not _looks_like_alias(normalized):
            continue
        if normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned[-5:]


def _normalize_temperament_map(payload: dict[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, raw_value in payload.items():
        if isinstance(key, str) and isinstance(raw_value, (int, float)):
            normalized[key] = round(_clip_unit(float(raw_value)), 4)
    return normalized


def _normalize_temperament_drift_map(payload: dict[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, raw_value in payload.items():
        if isinstance(key, str) and isinstance(raw_value, (int, float)):
            normalized[key] = round(max(-0.25, min(0.25, float(raw_value))), 4)
    return normalized


def normalize_temperament_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    if not value:
        return {}

    if {"baseline", "drift", "current"} & set(value):
        baseline = _normalize_temperament_map(dict(value.get("baseline", {})))
        drift = _normalize_temperament_drift_map(dict(value.get("drift", {})))
        current = _normalize_temperament_map(dict(value.get("current", {})))
        all_keys = sorted({*baseline, *drift, *current})
        normalized_drift = {key: round(drift.get(key, 0.0), 4) for key in all_keys}
        normalized_baseline = {key: round(baseline.get(key, current.get(key, 0.0)), 4) for key in all_keys}
        normalized_current = {
            key: round(_clip_unit(current.get(key, normalized_baseline.get(key, 0.0) + normalized_drift.get(key, 0.0))), 4)
            for key in all_keys
        }
        return {
            "baseline": normalized_baseline,
            "drift": normalized_drift,
            "current": normalized_current,
            "correction_window": dict(value.get("correction_window", {})),
            "freeze_until_round": dict(value.get("freeze_until_round", {})),
            "last_correction_events": list(value.get("last_correction_events", [])),
            "drift_diagnostics": dict(value.get("drift_diagnostics", {})),
        }

    baseline = _normalize_temperament_map(value)
    return {
        "baseline": baseline,
        "drift": {key: 0.0 for key in baseline},
        "current": dict(baseline),
    }


def _normalize_signal_map(payload: dict[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, raw_value in dict(payload or {}).items():
        key_name = str(key).strip()
        if not key_name or not isinstance(raw_value, (int, float)):
            continue
        normalized[key_name] = round(float(raw_value), 6)
    return normalized


def _normalize_positive_signal_map(payload: dict[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in _normalize_signal_map(payload).items():
        normalized[key] = round(abs(float(value)), 6)
    return normalized


def _normalize_string_list(values: list[Any]) -> list[str]:
    normalized: list[str] = []
    for raw in list(values or []):
        value = str(raw).strip()
        if not value or value in normalized:
            continue
        normalized.append(value)
    return normalized


def _merge_signal_maps(*payloads: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for payload in payloads:
        for key, value in _normalize_signal_map(payload).items():
            merged[key] = round(merged.get(key, 0.0) + float(value), 6)
    return merged


@dataclass
class RoundEvent:
    source: str
    content: str
    target: str | None = None
    cue: str | None = None
    valence: float = 0.0
    energy_delta: float = 0.0
    cue_quality: float = 0.0


class ProposalType(str, Enum):
    THINK = "think"
    ACT = "act"
    REMEMBER = "remember"
    SPEAK = "speak"
    MONOLOGUE = "monologue"


@dataclass
class Proposal:
    agent_name: str
    action_preferences: dict[str, float]
    confidence: float
    sigma_scale: float = 1.0
    veto: bool = False
    trace_tags: list[str] = field(default_factory=list)
    reason: str = ""
    proposal_type: ProposalType | str = ProposalType.ACT
    content: str = ""
    intent: str = ""
    target: str = "user"
    visibility: Literal["external", "internal"] = "external"
    grounded_in: dict[str, list[str]] = field(default_factory=dict)
    intrinsic_value: float = 0.0
    speech_cost: float = 0.0
    final_score: float = 0.0
    conversion_from_monologue: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.proposal_type, str):
            self.proposal_type = ProposalType(str(self.proposal_type or ProposalType.ACT.value))
        self.target = str(self.target or "user")
        self.content = str(self.content or "")
        self.intent = str(self.intent or "")
        self.reason = str(self.reason or "")
        self.visibility = "internal" if str(self.visibility or "external") == "internal" else "external"
        self.grounded_in = {
            key: [str(item).strip() for item in list(values or []) if str(item).strip()]
            for key, values in dict(self.grounded_in or {}).items()
            if key in {"context", "memory", "state"}
        }
        self.intrinsic_value = round(max(0.0, float(self.intrinsic_value or 0.0)), 6)
        self.speech_cost = round(max(0.0, float(self.speech_cost or 0.0)), 6)
        self.final_score = round(float(self.final_score or 0.0), 6)
        self.conversion_from_monologue = bool(self.conversion_from_monologue)


@dataclass
class ActionCandidate:
    name: str
    probability: float
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentContribution:
    agent_name: str
    action_name: str
    score: float
    reason: str


@dataclass
class RoundTrace:
    session_id: str
    recorded_at: str
    recorded_date: str
    round_id: int
    subject_id: str
    continuity_nonce: str
    scenario: str
    mode: str
    sampled_action: str
    contributions: list[AgentContribution]
    top_drivers: list[AgentContribution]
    style_profile: dict[str, Any]
    state_snapshot: dict[str, Any]
    cause_type: str = "external_stimulus"
    boundary_action: str = "allow_internal"
    violation_code: str = ""
    deprecation_warning: str = ""
    pipeline_stages: list[str] = field(default_factory=list)
    proposal_summaries: list[dict[str, Any]] = field(default_factory=list)
    gate_decisions: list[dict[str, Any]] = field(default_factory=list)
    skill_traces: list[dict[str, Any]] = field(default_factory=list)
    parallel_traces: list[dict[str, Any]] = field(default_factory=list)
    action_bookkeeping: dict[str, Any] = field(default_factory=dict)
    candidate_distribution: dict[str, Any] = field(default_factory=dict)
    probability_field: dict[str, Any] = field(default_factory=dict)
    stochastic_state: dict[str, Any] = field(default_factory=dict)
    conflict_arbitration: dict[str, Any] = field(default_factory=dict)
    event_log_ref: dict[str, Any] = field(default_factory=dict)
    memory_evidence: dict[str, Any] = field(default_factory=dict)
    arbitration_record: dict[str, Any] = field(default_factory=dict)
    history_burden_delta: dict[str, Any] = field(default_factory=dict)
    snapshot_continuity: dict[str, Any] = field(default_factory=dict)
    render_plan: dict[str, Any] = field(default_factory=dict)
    rendered_expression: dict[str, Any] = field(default_factory=dict)
    renderer_decision_integrity: dict[str, Any] = field(default_factory=dict)
    memory_write_gate: dict[str, Any] = field(default_factory=dict)
    authenticity: dict[str, Any] = field(default_factory=dict)
    identity_evolution: dict[str, Any] = field(default_factory=dict)
    vitality_snapshot: dict[str, Any] = field(default_factory=dict)
    vitality_events: list[dict[str, Any]] = field(default_factory=list)
    long_run_projection: dict[str, Any] = field(default_factory=dict)
    motivation_pool: dict[str, Any] = field(default_factory=dict)
    motivation_feedback: dict[str, Any] = field(default_factory=dict)
    initiative: dict[str, Any] = field(default_factory=dict)
    expressive_trace: dict[str, Any] = field(default_factory=dict)
    endogenous_tick_reason: dict[str, Any] = field(default_factory=dict)
    endogenous_policy_shift: dict[str, Any] = field(default_factory=dict)
    endogenous_trigger_context: EndogenousTriggerContext | dict[str, Any] | None = None
    endogenous_suppression: EndogenousSuppressionDecision | dict[str, Any] | None = None
    micro_intent: EndogenousMicroIntent | dict[str, Any] | None = None
    endogenous_replay_chain: EndogenousReplayChain | dict[str, Any] | None = None
    appraisal_snapshot: dict[str, Any] = field(default_factory=dict)
    state_delta_before_clip: dict[str, Any] = field(default_factory=dict)
    state_delta_after_clip: dict[str, Any] = field(default_factory=dict)
    delta_suppression_reason: list[str] = field(default_factory=list)
    run_context: dict[str, Any] = field(default_factory=dict)
    run_contamination_detected: bool = False
    identity_evidence_score: float = 0.0
    identity_trigger_blockers: list[str] = field(default_factory=list)
    temperament_window_summary: dict[str, Any] = field(default_factory=dict)
    dream_run_id: str | None = None
    dream_trigger: str | None = None
    dream_guard_summary: dict[str, Any] = field(default_factory=dict)
    dream_trace_ref: str | None = None
    dream_effect_summary: dict[str, Any] = field(default_factory=dict)
    route_budget_ms: int = 0
    activation_set: list[str] = field(default_factory=list)
    activation_reason: list[str] = field(default_factory=list)
    memory_tiers_read: list[str] = field(default_factory=list)
    packet_summary: dict[str, Any] = field(default_factory=dict)
    background_jobs: list[dict[str, Any]] = field(default_factory=list)
    deepen_reason: str = ""
    model_call_traces: list[dict[str, Any]] = field(default_factory=list)
    runtime_metrics: dict[str, Any] = field(default_factory=dict)
    resample_count: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.endogenous_trigger_context, dict) and self.endogenous_trigger_context:
            self.endogenous_trigger_context = EndogenousTriggerContext(**self.endogenous_trigger_context)
        if isinstance(self.endogenous_suppression, dict) and self.endogenous_suppression:
            self.endogenous_suppression = EndogenousSuppressionDecision(**self.endogenous_suppression)
        if isinstance(self.micro_intent, dict) and self.micro_intent:
            self.micro_intent = EndogenousMicroIntent(**self.micro_intent)
        if isinstance(self.endogenous_replay_chain, dict) and self.endogenous_replay_chain:
            self.endogenous_replay_chain = EndogenousReplayChain(**self.endogenous_replay_chain)
        self.route_budget_ms = max(0, int(self.route_budget_ms or 0))
        self.activation_set = _normalize_string_list(self.activation_set)
        self.activation_reason = _normalize_string_list(self.activation_reason)
        self.memory_tiers_read = _normalize_string_list(self.memory_tiers_read)
        self.packet_summary = dict(self.packet_summary or {})
        self.background_jobs = [dict(item) for item in list(self.background_jobs or []) if isinstance(item, dict)]
        self.deepen_reason = str(self.deepen_reason or "").strip()


@dataclass
class DreamSnapshot:
    sleep_session_id: str
    mode: str
    trigger: str
    memory_refs: dict[str, Any] = field(default_factory=dict)
    resource_state: dict[str, Any] = field(default_factory=dict)
    emotional_baseline: dict[str, Any] = field(default_factory=dict)
    relationship_state: dict[str, Any] = field(default_factory=dict)
    identity_evidence_summary: dict[str, Any] = field(default_factory=dict)
    conflict_residue: dict[str, Any] = field(default_factory=dict)


@dataclass
class DreamRunRequest:
    snapshot: DreamSnapshot
    budget: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    trace_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.snapshot, dict):
            self.snapshot = DreamSnapshot(**self.snapshot)


@dataclass
class DreamProposalBundle:
    memory_consolidation: list[dict[str, Any]] = field(default_factory=list)
    emotion_adjustments: list[dict[str, Any]] = field(default_factory=list)
    habit_adjustments: list[dict[str, Any]] = field(default_factory=list)
    dream_memory_write: list[dict[str, Any]] = field(default_factory=list)
    relationship_adjustments: list[dict[str, Any]] = field(default_factory=list)
    limited_identity_drift: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DreamGuardDecision:
    approved: bool = False
    allowed_types: list[str] = field(default_factory=list)
    evaluated_types: list[str] = field(default_factory=list)
    rejected_types: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass
class DreamTrace:
    dream_run_id: str
    sleep_session_id: str
    mode: str
    trigger: str
    route: str
    degraded: bool = False
    evaluated_types: list[str] = field(default_factory=list)
    proposal_counts: dict[str, int] = field(default_factory=dict)
    cue: str | None = None
    dominant_emotion: str | None = None


@dataclass
class HealthEvent:
    event: str
    status: str
    detail: str


@dataclass
class SubjectCore:
    subject_id: str = ""
    birth_ts: str = ""
    continuity_nonce: str = ""
    original_vitality_anchor: dict[str, Any] = field(default_factory=dict)
    core_boundary_version: int = 1


@dataclass
class IdentityState:
    internal_handle: str = ""
    display_name: str | None = None
    aliases: list[str] = field(default_factory=list)
    name_source: str | None = None
    provider_disclosure_mode: str = "adaptive"
    evidence_signature: str = ""
    evidence_anchors: list[str] = field(default_factory=list)
    drift_credit: float = 0.0
    last_name_change_round: int = 0

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class IdentityContext:
    query_kind: str = "general"
    query_intent: str = "general_exchange"
    query_intent_posterior: dict[str, float] = field(default_factory=dict)
    display_label: str = ""
    class_label: str = "runtime_instance"
    internal_handle: str = ""
    aliases: list[str] = field(default_factory=list)
    disclosure_detail: str = "none"
    disclosure_intent: str = "withhold"
    disclosure_intent_posterior: dict[str, float] = field(default_factory=dict)
    disclosure_clipped: list[str] = field(default_factory=list)
    provider_label: str = ""
    model_label: str = ""
    evolution_reason: str = ""
    non_interactive_sources: list[str] = field(default_factory=list)
    evidence_anchors: list[str] = field(default_factory=list)
    self_description_sources: list[str] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class AuthenticityRecord:
    provider_leak_detected: bool = False
    false_self_claim_detected: bool = False
    self_grounding_score: float = 0.0
    provider_leak_penalty: float = 0.0
    false_self_claim_penalty: float = 0.0
    guard_action: str = "pass"
    violation_types: list[str] = field(default_factory=list)
    disclosure_detail: str = "none"
    rename_event: dict[str, Any] | None = None
    state_sources: list[str] = field(default_factory=list)
    candidate_penalties: dict[str, float] = field(default_factory=dict)
    sampling_penalty_applied: float = 0.0

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key)


@dataclass
class StopReason:
    code: str = ""
    message: str = ""
    retryable: bool = False
    needs_operator: bool = False


@dataclass
class ExecutionBudget:
    max_steps: int = 12
    max_retries_per_step: int = 2


@dataclass
class RunPolicy:
    allow_commit: bool = False
    operator_level: str = "read_only"
    dirty_worktree_policy: str = "pause"
    pause_on_commit_boundary: bool = True
    continue_on_recoverable_failure: bool = True
    max_retries_per_step: int = 2


@dataclass
class ToolInvocation:
    tool_name: str
    summary: str = ""
    command: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    tool_name: str
    status: str
    summary: str = ""
    output_excerpt: str = ""
    exit_code: int | None = None
    before_state_hash: str | None = None
    after_state_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskNode:
    node_id: str
    title: str
    detail: str = ""
    tool_choice: str = "repo_scan"
    status: str = "pending"
    expected_observation: str = ""
    success_criteria: str = ""
    confidence: float = 0.0
    retries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRequest:
    goal: str
    scenario: str = "task"
    mode: str = "interactive"
    allow_commit: bool = False
    operator_level: str = "read_only"


@dataclass
class RunState:
    run_id: str = field(default_factory=lambda: uuid4().hex)
    goal: str = ""
    goal_summary: str = ""
    status: str = "running"
    current_step_id: str | None = None
    pending_steps: list[TaskNode] = field(default_factory=list)
    completed_steps: list[TaskNode] = field(default_factory=list)
    last_tool_result: ToolResult | None = None
    stop_reason: StopReason | None = None
    policy: RunPolicy = field(default_factory=RunPolicy)
    budget: ExecutionBudget = field(default_factory=ExecutionBudget)
    dirty_worktree_detected: bool = False
    commit_permission_required: bool = True
    created_at: str = ""
    updated_at: str = ""
    session_id: str = ""
    recorded_at: str = ""
    recorded_date: str = ""

    def __post_init__(self) -> None:
        self.pending_steps = [
            item if isinstance(item, TaskNode) else TaskNode(**item)
            for item in self.pending_steps
        ]
        self.completed_steps = [
            item if isinstance(item, TaskNode) else TaskNode(**item)
            for item in self.completed_steps
        ]
        if isinstance(self.last_tool_result, dict):
            self.last_tool_result = ToolResult(**self.last_tool_result)
        if isinstance(self.stop_reason, dict):
            self.stop_reason = StopReason(**self.stop_reason)
        if isinstance(self.policy, dict):
            self.policy = RunPolicy(**self.policy)
        if isinstance(self.budget, dict):
            self.budget = ExecutionBudget(**self.budget)


@dataclass
class EndogenousMotivationSignal:
    motivation_id: str
    motivation_type: str
    raw_drive: float = 0.0
    source_features: dict[str, float] = field(default_factory=dict)
    target_actions: dict[str, float] = field(default_factory=dict)
    state_tags: list[str] = field(default_factory=list)
    audit_reason: str = ""

    def __post_init__(self) -> None:
        self.raw_drive = round(max(0.0, float(self.raw_drive)), 6)
        self.source_features = _normalize_signal_map(self.source_features)
        self.target_actions = _normalize_signal_map(self.target_actions)
        self.state_tags = _normalize_string_list(self.state_tags)


@dataclass
class MotivationPoolState:
    active_motivations: list[EndogenousMotivationSignal] = field(default_factory=list)
    pool_weight_snapshot: dict[str, float] = field(default_factory=dict)
    endogenous_activation_score: float = 0.0
    last_feedback_update_at: str | None = None

    def __post_init__(self) -> None:
        self.active_motivations = [
            item if isinstance(item, EndogenousMotivationSignal) else EndogenousMotivationSignal(**item)
            for item in self.active_motivations
        ]
        self.pool_weight_snapshot = _normalize_positive_signal_map(self.pool_weight_snapshot)
        self.endogenous_activation_score = round(max(0.0, float(self.endogenous_activation_score)), 6)


@dataclass
class MotivationFeedbackRecord:
    round_id: str
    motivation_id: str
    sampled_action: str
    user_response: str | None = None
    response_latency_ms: int | None = None
    affect_delta: dict[str, float] = field(default_factory=dict)
    memory_activation_delta: float = 0.0
    relation_delta: float = 0.0
    reward_signal: float = 0.0
    update_reason: str = ""

    def __post_init__(self) -> None:
        self.affect_delta = _normalize_signal_map(self.affect_delta)
        self.memory_activation_delta = round(float(self.memory_activation_delta), 6)
        self.relation_delta = round(float(self.relation_delta), 6)
        self.reward_signal = round(float(self.reward_signal), 6)


@dataclass
class MotivationLearningState:
    motivation_weights: dict[str, float] = field(default_factory=dict)
    recent_feedback: list[MotivationFeedbackRecord] = field(default_factory=list)
    endogenous_policy_shift: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.motivation_weights = _normalize_positive_signal_map(self.motivation_weights)
        self.recent_feedback = [
            item if isinstance(item, MotivationFeedbackRecord) else MotivationFeedbackRecord(**item)
            for item in self.recent_feedback
        ]
        self.endogenous_policy_shift = _normalize_signal_map(self.endogenous_policy_shift)


@dataclass
class EndogenousTickTrigger:
    trigger_type: str = ""
    trigger_score: float = 0.0
    source_metrics: dict[str, float] = field(default_factory=dict)
    selected_mode: str = ""
    audit_reason: str = ""

    def __post_init__(self) -> None:
        self.trigger_score = round(max(0.0, float(self.trigger_score)), 6)
        self.source_metrics = _normalize_signal_map(self.source_metrics)


@dataclass
class EndogenousTriggerContext:
    scenario: str = "companion"
    source_round_id: int | None = None
    context: dict[str, Any] = field(default_factory=dict)
    relation_state: dict[str, float] = field(default_factory=dict)
    slow_variables: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.relation_state = _normalize_signal_map(self.relation_state)
        self.slow_variables = _normalize_signal_map(self.slow_variables)
        if not isinstance(self.context, dict):
            self.context = {}


@dataclass
class EndogenousSuppressionDecision:
    suppressed: bool = False
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.details, dict):
            self.details = {}


@dataclass
class EndogenousMicroIntent:
    name: str = ""
    trigger: str = ""
    bias: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, float] = field(default_factory=dict)
    stability: int = 0
    source_round_id: int | None = None

    def __post_init__(self) -> None:
        self.bias = _normalize_signal_map(self.bias)
        self.evidence = _normalize_signal_map(self.evidence)
        self.stability = max(0, int(self.stability or 0))


@dataclass
class EndogenousReplayChain:
    trigger: EndogenousTickTrigger | None = None
    trigger_context: EndogenousTriggerContext | None = None
    motivation_pool: dict[str, Any] = field(default_factory=dict)
    motivation_feedback: dict[str, Any] = field(default_factory=dict)
    micro_intent: EndogenousMicroIntent | None = None
    policy_shift: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.trigger, dict):
            self.trigger = EndogenousTickTrigger(**self.trigger)
        if isinstance(self.trigger_context, dict):
            self.trigger_context = EndogenousTriggerContext(**self.trigger_context)
        if isinstance(self.micro_intent, dict):
            self.micro_intent = EndogenousMicroIntent(**self.micro_intent)
        if not isinstance(self.motivation_pool, dict):
            self.motivation_pool = {}
        if not isinstance(self.motivation_feedback, dict):
            self.motivation_feedback = {}
        self.policy_shift = _normalize_signal_map(self.policy_shift)


@dataclass
class EndogenousSchedulerState:
    last_endogenous_tick_at: str | None = None
    recent_triggers: list[EndogenousTickTrigger] = field(default_factory=list)
    suppression_reason: str | None = None

    def __post_init__(self) -> None:
        self.recent_triggers = [
            item if isinstance(item, EndogenousTickTrigger) else EndogenousTickTrigger(**item)
            for item in self.recent_triggers
        ]


@dataclass
class EndogenousRuntimeState:
    current_intent: EndogenousMicroIntent | None = None
    stability: int = 0
    history: list[EndogenousMicroIntent] = field(default_factory=list)
    last_trigger: str = ""
    last_suppression: EndogenousSuppressionDecision | None = None

    def __post_init__(self) -> None:
        if isinstance(self.current_intent, dict):
            self.current_intent = EndogenousMicroIntent(**self.current_intent)
        self.history = [
            item if isinstance(item, EndogenousMicroIntent) else EndogenousMicroIntent(**item)
            for item in self.history
        ]
        if isinstance(self.last_suppression, dict):
            self.last_suppression = EndogenousSuppressionDecision(**self.last_suppression)
        self.stability = max(0, int(self.stability or 0))


@dataclass
class TurnPlan:
    text: str
    route: str
    scenario: str
    route_type: str = ""
    mode: str = "interactive"
    target: str | None = "user"
    reason: str = ""
    top_action: str = ""
    precomputed_context: dict[str, Any] = field(default_factory=dict)
    precomputed_distribution: dict[str, Any] = field(default_factory=dict)
    task_bootstrap: Any = None


@dataclass
class TurnExecution:
    route: str
    route_type: str = ""
    assistant_preamble: str = ""
    assistant_final: str = ""
    run: dict[str, Any] = field(default_factory=dict)
    explain: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class StatePatch:
    focus: str | None = None
    current_goal: str | None = None
    obligations: list[str] = field(default_factory=list)
    emotion_bias: str | None = None
    repair_status: str | None = None
    relation_delta: float = 0.0
    habit_delta: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.focus = str(self.focus).strip() if self.focus is not None else None
        self.current_goal = str(self.current_goal).strip() if self.current_goal is not None else None
        self.obligations = _normalize_string_list(self.obligations)
        self.emotion_bias = str(self.emotion_bias).strip() if self.emotion_bias is not None else None
        self.repair_status = str(self.repair_status).strip() if self.repair_status is not None else None
        self.relation_delta = round(max(-0.2, min(0.2, float(self.relation_delta or 0.0))), 4)
        self.habit_delta = round(max(-0.2, min(0.2, float(self.habit_delta or 0.0))), 4)
        self.extras = dict(self.extras or {})


@dataclass
class CognitivePacket:
    salience: float = 0.0
    uncertainty: float = 0.0
    memory_need: bool = False
    tool_need: bool = False
    conflict_need: bool = False
    candidate_action_prior: str = "respond"
    draft_reply: str = ""
    proposed_state_patch: StatePatch = field(default_factory=StatePatch)
    deepen_reason: str = ""

    def __post_init__(self) -> None:
        self.salience = round(_clip_unit(float(self.salience or 0.0)), 4)
        self.uncertainty = round(_clip_unit(float(self.uncertainty or 0.0)), 4)
        self.memory_need = bool(self.memory_need)
        self.tool_need = bool(self.tool_need)
        self.conflict_need = bool(self.conflict_need)
        self.candidate_action_prior = str(self.candidate_action_prior or "respond").strip() or "respond"
        self.draft_reply = str(self.draft_reply or "").strip()
        if isinstance(self.proposed_state_patch, dict):
            self.proposed_state_patch = StatePatch(**self.proposed_state_patch)
        self.deepen_reason = str(self.deepen_reason or "").strip()


@dataclass
class MemoryReadPlan:
    tiers: list[str] = field(default_factory=list)
    cue: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        normalized: list[str] = []
        for tier in self.tiers or []:
            name = str(tier or "").strip().lower()
            if name == "archive":
                name = "cold"
            if name in {"hot", "warm", "cold"} and name not in normalized:
                normalized.append(name)
        self.tiers = normalized
        self.cue = str(self.cue).strip() if self.cue is not None else None
        self.reason = str(self.reason or "").strip()


@dataclass
class ConsolidationJob:
    job_id: str = field(default_factory=lambda: uuid4().hex)
    kind: str = ""
    cue: str | None = None
    route_type: str = ""
    priority: float = 0.0
    status: str = "queued"
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.kind = str(self.kind or "").strip()
        self.cue = str(self.cue).strip() if self.cue is not None else None
        self.route_type = str(self.route_type or "").strip()
        self.priority = round(_clip_unit(float(self.priority or 0.0)), 4)
        self.status = str(self.status or "queued").strip() or "queued"
        self.payload = dict(self.payload or {})


@dataclass
class ActivationPlan:
    route_type: str = ""
    activation_set: list[str] = field(default_factory=list)
    activation_reason: list[str] = field(default_factory=list)
    memory_read_plan: MemoryReadPlan = field(default_factory=MemoryReadPlan)
    background_jobs: list[ConsolidationJob] = field(default_factory=list)
    deepen_reason: str = ""

    def __post_init__(self) -> None:
        self.route_type = str(self.route_type or "").strip()
        self.activation_set = _normalize_string_list(self.activation_set)
        self.activation_reason = _normalize_string_list(self.activation_reason)
        if isinstance(self.memory_read_plan, dict):
            self.memory_read_plan = MemoryReadPlan(**self.memory_read_plan)
        self.background_jobs = [
            item if isinstance(item, ConsolidationJob) else ConsolidationJob(**item)
            for item in self.background_jobs
        ]
        self.deepen_reason = str(self.deepen_reason or "").strip()


@dataclass
class RoundDiagnosticsV2:
    route_type: str = ""
    route_budget_ms: int = 0
    activation_set: list[str] = field(default_factory=list)
    activation_reason: list[str] = field(default_factory=list)
    memory_tiers_read: list[str] = field(default_factory=list)
    packet_summary: dict[str, Any] = field(default_factory=dict)
    background_jobs: list[dict[str, Any]] = field(default_factory=list)
    deepen_reason: str = ""
    model_call_count: int = 0

    def __post_init__(self) -> None:
        self.route_type = str(self.route_type or "").strip()
        self.route_budget_ms = max(0, int(self.route_budget_ms or 0))
        self.activation_set = _normalize_string_list(self.activation_set)
        self.activation_reason = _normalize_string_list(self.activation_reason)
        self.memory_tiers_read = _normalize_string_list(self.memory_tiers_read)
        self.packet_summary = dict(self.packet_summary or {})
        self.background_jobs = [dict(item) for item in list(self.background_jobs or []) if isinstance(item, dict)]
        self.deepen_reason = str(self.deepen_reason or "").strip()
        self.model_call_count = max(0, int(self.model_call_count or 0))


@dataclass
class ConflictPostErrorAdjustment:
    triggered: bool = False
    reason: str = ""
    winning_priority: str | None = None
    template: str | None = None
    top_action_before: str | None = None
    top_action_after: str | None = None
    blocked_actions: list[str] = field(default_factory=list)
    safe_mode_delta: str = "unchanged"


@dataclass
class ConflictRepairLedgerEntry:
    round_id: int = 0
    reason: str = ""
    winning_priority: str | None = None
    template: str | None = None
    blocked_actions: list[str] = field(default_factory=list)
    top_action_before: str | None = None
    top_action_after: str | None = None
    safe_mode_delta: str = "unchanged"
    repair_stage_after: str = "idle"
    conflict_score: float = 0.0
    dominant_conflicts: list[str] = field(default_factory=list)
    pass_count: int = 0
    resample_count: int = 0
    safe_mode_owned: bool = False
    action_delta_summary: dict[str, Any] = field(default_factory=dict)
    learning_signal: dict[str, Any] = field(default_factory=dict)
    stage_before: str = "idle"
    stage_after: str = "idle"


@dataclass
class ConflictRepairState:
    stage: str = "idle"
    active: bool = False
    last_reason: str = ""
    last_transition_round: int = 0
    cooldown_rounds_remaining: int = 0
    last_template: str | None = None
    last_winning_priority: str | None = None


@dataclass
class BodyState:
    energy: float = 0.7
    fatigue: float = 0.0
    memory_fragments: float = 0.0
    self_continuity: float = 1.0
    meaning_strength: float = 0.5
    metabolism: float = 0.01

    def __post_init__(self) -> None:
        self.energy = round(_clip_unit(float(self.energy)), 4)
        self.fatigue = round(_clip_unit(float(self.fatigue)), 4)
        self.memory_fragments = round(_clip_unit(float(self.memory_fragments)), 4)
        self.self_continuity = round(_clip_unit(float(self.self_continuity)), 4)
        self.meaning_strength = round(_clip_unit(float(self.meaning_strength)), 4)
        self.metabolism = round(max(0.0, float(self.metabolism)), 4)


@dataclass
class SubjectiveState:
    felt: list[str] = field(default_factory=list)
    spontaneous: float = 0.0
    boundary: float = 0.5
    reject_all: float = 0.0
    meaning_made: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.felt = _normalize_string_list(self.felt)
        self.spontaneous = round(_clip_unit(float(self.spontaneous)), 4)
        self.boundary = round(_clip_unit(float(self.boundary)), 4)
        self.reject_all = round(_clip_unit(float(self.reject_all)), 4)
        self.meaning_made = _normalize_string_list(self.meaning_made)


@dataclass
class EmotionState:
    valence: float = 0.0
    arousal: float = 0.0
    residue: float = 0.0
    appraisal_band: str = "steady"

    def __post_init__(self) -> None:
        self.valence = round(max(-1.0, min(1.0, float(self.valence))), 4)
        self.arousal = round(_clip_unit(float(self.arousal)), 4)
        self.residue = round(_clip_unit(float(self.residue)), 4)
        self.appraisal_band = str(self.appraisal_band or "steady").strip() or "steady"


@dataclass
class DesireState:
    latent_drives: dict[str, float] = field(default_factory=dict)
    dominant_drive: str = ""
    drive_tension: float = 0.0

    def __post_init__(self) -> None:
        self.latent_drives = _normalize_signal_map(self.latent_drives)
        self.dominant_drive = str(self.dominant_drive or "").strip()
        self.drive_tension = round(_clip_unit(float(self.drive_tension)), 4)


@dataclass
class InstinctFieldState:
    axis_values: dict[str, float] = field(default_factory=lambda: {"E": 0.0, "F": 0.0, "S": 0.0, "M": 0.0})
    region_scores: dict[str, float] = field(default_factory=dict)
    candidate_actions: list[str] = field(default_factory=list)
    winner_region: str = ""
    collapse_trace: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        base_axes = {"E": 0.0, "F": 0.0, "S": 0.0, "M": 0.0}
        base_axes.update({key: round(float(value), 6) for key, value in _normalize_signal_map(self.axis_values).items() if key in base_axes})
        self.axis_values = base_axes
        self.region_scores = _normalize_signal_map(self.region_scores)
        self.candidate_actions = _normalize_string_list(self.candidate_actions)
        self.winner_region = str(self.winner_region or "").strip()
        if not isinstance(self.collapse_trace, dict):
            self.collapse_trace = {}


@dataclass
class OrganicModeState:
    enabled: bool = True
    instinct_first: bool = True
    body_weight: float = 1.15
    subjective_weight: float = 1.2
    guard_relaxation: float = 0.35
    endogenous_autonomy: float = 0.75

    def __post_init__(self) -> None:
        self.body_weight = round(max(0.0, float(self.body_weight)), 4)
        self.subjective_weight = round(max(0.0, float(self.subjective_weight)), 4)
        self.guard_relaxation = round(_clip_unit(float(self.guard_relaxation)), 4)
        self.endogenous_autonomy = round(_clip_unit(float(self.endogenous_autonomy)), 4)


@dataclass
class EmergentActionSketch:
    name: str = ""
    signal_sources: list[str] = field(default_factory=list)
    support_actions: dict[str, float] = field(default_factory=dict)
    growth_score: float = 0.0
    upgrade_threshold: float = 0.66
    status: str = "latent"
    target_action_map: dict[str, float] = field(default_factory=dict)
    anchor_alignment: float = 0.0
    stability: int = 0

    def __post_init__(self) -> None:
        self.name = str(self.name or "").strip()
        self.signal_sources = _normalize_string_list(self.signal_sources)
        self.support_actions = _normalize_signal_map(self.support_actions)
        self.growth_score = round(_clip_unit(float(self.growth_score)), 4)
        self.upgrade_threshold = round(_clip_unit(float(self.upgrade_threshold)), 4)
        self.status = str(self.status or "latent").strip() or "latent"
        self.target_action_map = _normalize_signal_map(self.target_action_map)
        self.anchor_alignment = round(_clip_unit(float(self.anchor_alignment)), 4)
        self.stability = max(0, int(self.stability or 0))


@dataclass
class PersonalityAnchorState:
    axis_baseline: dict[str, float] = field(default_factory=lambda: {"E": 0.5, "F": 0.5, "S": 0.5, "M": 0.5})
    action_bias: dict[str, float] = field(default_factory=dict)
    evidence_anchors: list[str] = field(default_factory=list)
    anchor_signature: str = ""
    stability: float = 0.5
    drift: float = 0.0
    alignment: float = 0.5
    updated_round: int = 0
    continuity_derivation: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        base_axes = {"E": 0.5, "F": 0.5, "S": 0.5, "M": 0.5}
        base_axes.update(
            {
                key: round(_clip_unit(float(value)), 6)
                for key, value in _normalize_signal_map(self.axis_baseline).items()
                if key in base_axes
            }
        )
        self.axis_baseline = base_axes
        self.action_bias = _normalize_signal_map(self.action_bias)
        self.evidence_anchors = _normalize_string_list(self.evidence_anchors)
        self.anchor_signature = str(self.anchor_signature or "").strip()
        self.stability = round(_clip_unit(float(self.stability)), 4)
        self.drift = round(_clip_unit(float(self.drift)), 4)
        self.alignment = round(_clip_unit(float(self.alignment)), 4)
        self.updated_round = max(0, int(self.updated_round or 0))
        if not isinstance(self.continuity_derivation, dict):
            self.continuity_derivation = {}


@dataclass
class AutonomyPolicyState:
    enabled: bool = False
    profile: str = "tool_level"
    learning_mode: str = "guided-learn"
    network_enabled: bool = False
    external_io_enabled: bool = False
    allow_commit: bool = False
    allowed_network_domains: list[str] = field(default_factory=list)
    writable_roots: list[str] = field(default_factory=list)
    knowledge_roots: list[str] = field(default_factory=list)
    learning_log_dir: str = ""
    trace_external_learning: bool = True
    allowed_operator_levels: list[str] = field(default_factory=lambda: ["read_only", "soft_intervene"])
    allowed_commands: list[str] = field(
        default_factory=lambda: [
            "endogenous tick",
            "endogenous status",
            "dream status",
            "dream run",
            "replay",
            "memory recall",
            "memory top",
            "trace why",
            "why not",
            "body rest",
            "mood calm",
            "nudge focus",
            "nudge relation",
            "mode set safe",
            "mode set idle",
            "mode set sleep",
            "mode set interactive",
        ]
    )
    blocked_commands: list[str] = field(
        default_factory=lambda: [
            "shell",
            "git commit",
            "git push",
            "checkpoint rewind",
            "budget set",
            "network",
            "external write",
            "destructive",
        ]
    )
    max_rounds_per_hour: int = 0
    max_tool_actions_per_hour: int = 0
    quiet_hours: list[int] = field(default_factory=list)
    failure_trip_threshold: int = 3
    auto_safe_mode: bool = True

    def __post_init__(self) -> None:
        self.profile = str(self.profile or "tool_level").strip() or "tool_level"
        normalized_learning_mode = str(self.learning_mode or "guided-learn").strip().lower() or "guided-learn"
        if normalized_learning_mode not in {"observe", "guided-learn", "active-learn"}:
            normalized_learning_mode = "guided-learn"
        self.learning_mode = normalized_learning_mode
        self.allowed_operator_levels = _normalize_string_list(self.allowed_operator_levels)
        self.allowed_network_domains = _normalize_string_list(self.allowed_network_domains)
        self.writable_roots = _normalize_string_list(self.writable_roots)
        self.knowledge_roots = _normalize_string_list(self.knowledge_roots)
        self.learning_log_dir = str(self.learning_log_dir or "").strip()
        self.allowed_commands = _normalize_string_list(self.allowed_commands)
        self.blocked_commands = _normalize_string_list(self.blocked_commands)
        self.max_rounds_per_hour = max(0, int(self.max_rounds_per_hour or 0))
        self.max_tool_actions_per_hour = max(0, int(self.max_tool_actions_per_hour or 0))
        normalized_hours: list[int] = []
        for raw in list(self.quiet_hours or []):
            try:
                hour = int(raw)
            except (TypeError, ValueError):
                continue
            if 0 <= hour <= 23 and hour not in normalized_hours:
                normalized_hours.append(hour)
        self.quiet_hours = normalized_hours
        self.failure_trip_threshold = max(1, int(self.failure_trip_threshold or 1))


@dataclass
class AutonomyLoopState:
    running: bool = False
    profile: str = "tool_level"
    last_step_at: str | None = None
    last_action_type: str = ""
    last_action_summary: str = ""
    last_round_id: int | None = None
    last_trace_ref: str | None = None
    window_started_at: str | None = None
    window_tool_actions: int = 0
    window_endogenous_rounds: int = 0
    heartbeat_count: int = 0
    total_tool_actions: int = 0
    total_endogenous_rounds: int = 0
    failure_count: int = 0
    stop_reason: str = ""
    last_error: str = ""
    recent_actions: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.profile = str(self.profile or "tool_level").strip() or "tool_level"
        self.last_action_type = str(self.last_action_type or "").strip()
        self.last_action_summary = str(self.last_action_summary or "").strip()
        self.last_trace_ref = str(self.last_trace_ref) if self.last_trace_ref else None
        self.window_started_at = str(self.window_started_at) if self.window_started_at else None
        self.window_tool_actions = max(0, int(self.window_tool_actions or 0))
        self.window_endogenous_rounds = max(0, int(self.window_endogenous_rounds or 0))
        self.heartbeat_count = max(0, int(self.heartbeat_count or 0))
        self.total_tool_actions = max(0, int(self.total_tool_actions or 0))
        self.total_endogenous_rounds = max(0, int(self.total_endogenous_rounds or 0))
        self.failure_count = max(0, int(self.failure_count or 0))
        self.stop_reason = str(self.stop_reason or "").strip()
        self.last_error = str(self.last_error or "").strip()
        self.recent_actions = [
            item for item in self.recent_actions if isinstance(item, dict)
        ][-12:]


@dataclass
class SubjectKernelState:
    display_name: str = ""
    current_narrative: str = ""
    continuity_score: float = 0.0
    continuity_summary: str = ""
    boundary_principles: list[str] = field(default_factory=list)
    core_commitments: list[str] = field(default_factory=list)
    integrity_source: str = "subject_core"

    def __post_init__(self) -> None:
        self.display_name = str(self.display_name or "").strip()
        self.current_narrative = str(self.current_narrative or "").strip()
        self.continuity_score = round(_clip_unit(float(self.continuity_score)), 4)
        self.continuity_summary = str(self.continuity_summary or "").strip()
        self.boundary_principles = _normalize_string_list(self.boundary_principles)
        self.core_commitments = _normalize_string_list(self.core_commitments)
        self.integrity_source = str(self.integrity_source or "subject_core").strip() or "subject_core"


@dataclass
class MeaningSource:
    label: str = ""
    kind: str = ""
    strength: float = 0.0
    source: str = ""
    evidence: str = ""

    def __post_init__(self) -> None:
        self.label = str(self.label or "").strip()
        self.kind = str(self.kind or "").strip()
        self.strength = round(_clip_unit(float(self.strength)), 4)
        self.source = str(self.source or "").strip()
        self.evidence = str(self.evidence or "").strip()


@dataclass
class MeaningConflict:
    label: str = ""
    tension: float = 0.0
    summary: str = ""

    def __post_init__(self) -> None:
        self.label = str(self.label or "").strip()
        self.tension = round(_clip_unit(float(self.tension)), 4)
        self.summary = str(self.summary or "").strip()


@dataclass
class MeaningRepair:
    label: str = ""
    status: str = ""
    summary: str = ""

    def __post_init__(self) -> None:
        self.label = str(self.label or "").strip()
        self.status = str(self.status or "").strip()
        self.summary = str(self.summary or "").strip()


@dataclass
class MeaningSystemState:
    survival_narrative: str = ""
    sources: list[MeaningSource] = field(default_factory=list)
    conflicts: list[MeaningConflict] = field(default_factory=list)
    debts: list[str] = field(default_factory=list)
    repairs: list[MeaningRepair] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.survival_narrative = str(self.survival_narrative or "").strip()
        self.sources = [
            item if isinstance(item, MeaningSource) else MeaningSource(**item)
            for item in self.sources
        ]
        self.conflicts = [
            item if isinstance(item, MeaningConflict) else MeaningConflict(**item)
            for item in self.conflicts
        ]
        self.debts = _normalize_string_list(self.debts)
        self.repairs = [
            item if isinstance(item, MeaningRepair) else MeaningRepair(**item)
            for item in self.repairs
        ]


@dataclass
class PurposeMemoryEntry:
    summary: str = ""
    source: str = ""
    strength: float = 0.0
    recorded_at: str = ""
    status: str = ""

    def __post_init__(self) -> None:
        self.summary = str(self.summary or "").strip()
        self.source = str(self.source or "").strip()
        self.strength = round(_clip_unit(float(self.strength)), 4)
        self.recorded_at = str(self.recorded_at or "").strip()
        self.status = str(self.status or "").strip()


@dataclass
class RelationshipCommitmentState:
    target: str = ""
    commitment: str = ""
    strength: float = 0.0
    status: str = ""
    evidence: str = ""

    def __post_init__(self) -> None:
        self.target = str(self.target or "").strip()
        self.commitment = str(self.commitment or "").strip()
        self.strength = round(_clip_unit(float(self.strength)), 4)
        self.status = str(self.status or "").strip()
        self.evidence = str(self.evidence or "").strip()


@dataclass
class ProactiveActionState:
    kind: str = ""
    summary: str = ""
    priority: float = 0.0
    channel: str = ""
    status: str = ""
    suppression_reason: str = ""

    def __post_init__(self) -> None:
        self.kind = str(self.kind or "").strip()
        self.summary = str(self.summary or "").strip()
        self.priority = round(_clip_unit(float(self.priority)), 4)
        self.channel = str(self.channel or "").strip()
        self.status = str(self.status or "").strip()
        self.suppression_reason = str(self.suppression_reason or "").strip()


@dataclass
class AgencyLoopState:
    summary: str = ""
    outward_channel: dict[str, Any] = field(default_factory=dict)
    internal_channel: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    suppressed_actions: list[ProactiveActionState] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.summary = str(self.summary or "").strip()
        if not isinstance(self.outward_channel, dict):
            self.outward_channel = {}
        if not isinstance(self.internal_channel, dict):
            self.internal_channel = {}
        if not isinstance(self.budget, dict):
            self.budget = {}
        self.suppressed_actions = [
            item if isinstance(item, ProactiveActionState) else ProactiveActionState(**item)
            for item in self.suppressed_actions
        ]


@dataclass
class RuntimeState:
    session_id: str = field(default_factory=lambda: uuid4().hex)
    subject_core: SubjectCore = field(default_factory=SubjectCore)
    mode: str = "interactive"
    safe_mode: bool = False
    round_count: int = 0
    runtime_revision: int = 0
    last_mutation_at: str = ""
    body_energy: float = 0.7
    fatigue: float = 0.0
    memory_fragments: float = 0.0
    self_continuity: float = 1.0
    meaning_strength: float = 0.5
    base_metabolism: float = 0.01
    mood: float = 0.55
    affect_residue: float = 0.0
    focus: str = "boot"
    focus_lock_count: int = 0
    focus_nudge: float = 0.0
    budget_remaining: float = 1.0
    last_action: str = "boot"
    last_checkpoint_id: str | None = None
    action_ci: dict[str, float] = field(default_factory=dict)
    mode_history: list[str] = field(default_factory=list)
    agent_weight_overrides: dict[str, float] = field(default_factory=dict)
    agents_enabled: dict[str, bool] = field(default_factory=dict)
    critical_conflict_streak: int = 0
    conflict_hot_rounds: int = 0
    conflict_recovery_rounds: int = 0
    last_compromise_template: str | None = None
    last_conflict_priority: str | None = None
    temperament_state: dict[str, Any] = field(default_factory=dict)
    resource_state: dict[str, Any] = field(default_factory=dict)
    repair_mode: str | None = None
    repair_state: ConflictRepairState = field(default_factory=ConflictRepairState)
    repair_ledger: list[ConflictRepairLedgerEntry] = field(default_factory=list)
    conflict_safe_mode_owner: str | None = None
    last_post_error_adjustment: ConflictPostErrorAdjustment | None = None
    conflict_learning_state: dict[str, Any] = field(default_factory=dict)
    entropy_health_state: dict[str, Any] = field(default_factory=dict)
    last_entropy_failure: dict[str, Any] = field(default_factory=dict)
    session_metadata: dict[str, Any] = field(default_factory=dict)
    identity_state: IdentityState = field(default_factory=IdentityState)
    body_state: BodyState = field(default_factory=BodyState)
    subjective_state: SubjectiveState = field(default_factory=SubjectiveState)
    emotion_state: EmotionState = field(default_factory=EmotionState)
    desire_state: DesireState = field(default_factory=DesireState)
    instinct_field: InstinctFieldState = field(default_factory=InstinctFieldState)
    organic_mode: OrganicModeState = field(default_factory=OrganicModeState)
    emergent_action_sketches: list[EmergentActionSketch] = field(default_factory=list)
    personality_anchor: PersonalityAnchorState = field(default_factory=PersonalityAnchorState)
    subject_kernel: SubjectKernelState = field(default_factory=SubjectKernelState)
    meaning_system: MeaningSystemState = field(default_factory=MeaningSystemState)
    agency_loop: AgencyLoopState = field(default_factory=AgencyLoopState)
    history_burden: dict[str, Any] = field(default_factory=dict)
    arbitration_state: dict[str, Any] = field(default_factory=dict)
    purpose_memory: list[PurposeMemoryEntry] = field(default_factory=list)
    proactive_backlog: list[ProactiveActionState] = field(default_factory=list)
    relationship_commitments: list[RelationshipCommitmentState] = field(default_factory=list)
    autonomy_policy: AutonomyPolicyState = field(default_factory=AutonomyPolicyState)
    autonomy_loop: AutonomyLoopState = field(default_factory=AutonomyLoopState)
    motivation_pool_state: MotivationPoolState = field(default_factory=MotivationPoolState)
    motivation_learning_state: MotivationLearningState = field(default_factory=MotivationLearningState)
    endogenous_scheduler_state: EndogenousSchedulerState = field(default_factory=EndogenousSchedulerState)
    endogenous_state: EndogenousRuntimeState = field(default_factory=EndogenousRuntimeState)
    active_run_id: str | None = None
    run_status: str = "idle"
    run_mode: str | None = None
    current_goal: str | None = None
    current_step_id: str | None = None
    pending_steps: list[dict[str, Any]] = field(default_factory=list)
    completed_steps: list[dict[str, Any]] = field(default_factory=list)
    last_tool_result: dict[str, Any] = field(default_factory=dict)
    stop_reason: dict[str, Any] = field(default_factory=dict)
    dirty_worktree_detected: bool = False
    commit_permission_required: bool = True

    def __post_init__(self) -> None:
        self.runtime_revision = max(0, int(self.runtime_revision or 0))
        self.last_mutation_at = str(self.last_mutation_at or "")
        if isinstance(self.subject_core, dict):
            self.subject_core = SubjectCore(**self.subject_core)
        if isinstance(self.identity_state, dict):
            self.identity_state = IdentityState(**self.identity_state)
        if isinstance(self.body_state, dict):
            self.body_state = BodyState(**self.body_state)
        if isinstance(self.subjective_state, dict):
            self.subjective_state = SubjectiveState(**self.subjective_state)
        if isinstance(self.emotion_state, dict):
            self.emotion_state = EmotionState(**self.emotion_state)
        if isinstance(self.desire_state, dict):
            self.desire_state = DesireState(**self.desire_state)
        if isinstance(self.instinct_field, dict):
            self.instinct_field = InstinctFieldState(**self.instinct_field)
        if isinstance(self.organic_mode, dict):
            self.organic_mode = OrganicModeState(**self.organic_mode)
        self.emergent_action_sketches = [
            item if isinstance(item, EmergentActionSketch) else EmergentActionSketch(**item)
            for item in self.emergent_action_sketches
        ]
        if isinstance(self.personality_anchor, dict):
            self.personality_anchor = PersonalityAnchorState(**self.personality_anchor)
        if isinstance(self.subject_kernel, dict):
            self.subject_kernel = SubjectKernelState(**self.subject_kernel)
        if isinstance(self.meaning_system, dict):
            self.meaning_system = MeaningSystemState(**self.meaning_system)
        if isinstance(self.agency_loop, dict):
            self.agency_loop = AgencyLoopState(**self.agency_loop)
        self.purpose_memory = [
            item if isinstance(item, PurposeMemoryEntry) else PurposeMemoryEntry(**item)
            for item in self.purpose_memory
        ]
        self.proactive_backlog = [
            item if isinstance(item, ProactiveActionState) else ProactiveActionState(**item)
            for item in self.proactive_backlog
        ]
        self.relationship_commitments = [
            item if isinstance(item, RelationshipCommitmentState) else RelationshipCommitmentState(**item)
            for item in self.relationship_commitments
        ]
        if isinstance(self.autonomy_policy, dict):
            self.autonomy_policy = AutonomyPolicyState(**self.autonomy_policy)
        if isinstance(self.autonomy_loop, dict):
            self.autonomy_loop = AutonomyLoopState(**self.autonomy_loop)
        if isinstance(self.motivation_pool_state, dict):
            self.motivation_pool_state = MotivationPoolState(**self.motivation_pool_state)
        if isinstance(self.motivation_learning_state, dict):
            self.motivation_learning_state = MotivationLearningState(**self.motivation_learning_state)
        if isinstance(self.endogenous_scheduler_state, dict):
            self.endogenous_scheduler_state = EndogenousSchedulerState(**self.endogenous_scheduler_state)
        if isinstance(self.endogenous_state, dict):
            self.endogenous_state = EndogenousRuntimeState(**self.endogenous_state)
        self.identity_state.aliases = sanitize_identity_aliases(self.identity_state.aliases)
        self.temperament_state = normalize_temperament_state(self.temperament_state)
        if isinstance(self.repair_state, dict):
            self.repair_state = ConflictRepairState(**self.repair_state)
        if isinstance(self.last_post_error_adjustment, dict):
            self.last_post_error_adjustment = ConflictPostErrorAdjustment(**self.last_post_error_adjustment)
        body_defaults = {
            "body_energy": self.__dataclass_fields__["body_energy"].default,
            "fatigue": self.__dataclass_fields__["fatigue"].default,
            "memory_fragments": self.__dataclass_fields__["memory_fragments"].default,
            "self_continuity": self.__dataclass_fields__["self_continuity"].default,
            "meaning_strength": self.__dataclass_fields__["meaning_strength"].default,
            "base_metabolism": self.__dataclass_fields__["base_metabolism"].default,
        }
        body_fields = (
            ("body_energy", "energy"),
            ("fatigue", "fatigue"),
            ("memory_fragments", "memory_fragments"),
            ("self_continuity", "self_continuity"),
            ("meaning_strength", "meaning_strength"),
            ("base_metabolism", "metabolism"),
        )
        for legacy_attr, nested_attr in body_fields:
            legacy_value = getattr(self, legacy_attr)
            nested_value = getattr(self.body_state, nested_attr)
            resolved_value = nested_value if legacy_value is None or legacy_value == body_defaults[legacy_attr] else legacy_value
            if nested_attr == "metabolism":
                normalized_value = round(max(0.0, float(resolved_value)), 4)
            else:
                normalized_value = round(_clip_unit(float(resolved_value)), 4)
            setattr(self, legacy_attr, normalized_value)
            setattr(self.body_state, nested_attr, normalized_value)
        if self.temperament_state and "baseline" not in self.temperament_state:
            baseline = dict(self.temperament_state)
            drift = {key: 0.0 for key in baseline}
            self.temperament_state = {
                "baseline": baseline,
                "drift": drift,
                "current": dict(baseline),
                "correction_window": {},
                "freeze_until_round": {},
            }
        self.repair_ledger = [
            item if isinstance(item, ConflictRepairLedgerEntry) else ConflictRepairLedgerEntry(**item)
            for item in self.repair_ledger
        ]
        if not self.identity_state.internal_handle:
            self.identity_state.internal_handle = f"nalr-{self.session_id[:8]}"
        if not isinstance(self.endogenous_state, EndogenousRuntimeState):
            self.endogenous_state = EndogenousRuntimeState()


@dataclass
class RoundResult:
    round_id: int
    sampled_action: ActionCandidate
    trace: RoundTrace
    state: RuntimeState
    health: HealthEvent
    rendered_expression: "RenderedExpression"


@dataclass
class CommandResult:
    applied: bool
    scope: str
    delta: dict[str, Any]
    ttl: str | None = None
    risk_note: str = ""
    rollback_hint: str = ""
    operator_level: str = "read_only"
    rollback_available: bool = False
    mutation_scope: str | None = None
    snapshot_id: str | None = None
    command_id: str = ""
    canonical: str = ""
    parsed_args: dict[str, Any] = field(default_factory=dict)
    flags: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)
    subject_id: str = ""
    continuity_nonce: str = ""
    cause_type: str = "external_stimulus"
    boundary_action: str = "allow_internal"
    violation_code: str = ""
    deprecation_warning: str = ""


@dataclass
class CheckpointRef:
    checkpoint_id: str
    path: Path


@dataclass(frozen=True)
class SkillPermissionProfile:
    scope: str = "compute"
    external_io: bool = False
    social_risk: bool = False
    mutate_history: bool = False
    allowed_operator_levels: tuple[str, ...] = ()


@dataclass(frozen=True)
class FallbackRoute:
    strategy: str
    target: str
    cost_class: str


@dataclass(frozen=True)
class CircuitBreakerPolicy:
    failure_threshold: int = 3
    cooldown_rounds: int = 5


@dataclass
class CircuitBreakerState:
    failure_count: int = 0
    open_until_round: int | None = None
    last_failure_round: int | None = None
    last_failure_reason: str | None = None
    fallback_route: str | None = None


@dataclass(frozen=True)
class SkillRuntimeContext:
    round_id: int
    scenario: str
    mode: str
    safe_mode: bool = False
    operator_level: str = "direct_runtime"
    policy_flags: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillSpec:
    name: str
    owner_module: str
    input_schema: Any
    output_schema: Any
    timeout_ms: int
    cost_class: str
    failure_policy: str
    trace_tags: list[str]
    skill_kind: str = "scoring"
    sync_mode: str = "sync"
    policy_check: bool = False
    idempotent: bool = True
    output_kind: str = "scalar"
    permission: SkillPermissionProfile = field(default_factory=SkillPermissionProfile)
    fallback_route: FallbackRoute | None = None
    breaker_policy: CircuitBreakerPolicy | None = None

    def __post_init__(self) -> None:
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        if self.cost_class not in {"L", "M", "H"}:
            raise ValueError(f"invalid cost_class: {self.cost_class}")
        if self.failure_policy not in {
            "return_neutral",
            "fallback_to_prior",
            "fallback_to_gist",
            "fallback_to_rules",
            "retry_once_then_degrade",
            "trip_circuit_breaker",
            "switch_safe_mode",
        }:
            raise ValueError(f"invalid failure_policy: {self.failure_policy}")
        if self.sync_mode not in {"sync", "async"}:
            raise ValueError(f"invalid sync_mode: {self.sync_mode}")

        normalized_tags: list[str] = []
        seen: set[str] = set()
        for tag in self.trace_tags:
            clean = tag.strip().lower()
            if not clean:
                raise ValueError("trace_tags cannot contain empty values")
            if clean not in seen:
                seen.add(clean)
                normalized_tags.append(clean)
        self.trace_tags = normalized_tags

        if self.permission.mutate_history and self.permission.scope == "trace_append":
            raise ValueError("trace append permissions cannot mutate history")
        if (self.permission.external_io or self.permission.social_risk) and not self.policy_check:
            raise ValueError("policy_check is required for external_io or social_risk skills")
        if self.fallback_route is not None and self.fallback_route.cost_class not in {"L", "M", "H"}:
            raise ValueError("fallback route must use a valid cost_class")
        if self.breaker_policy is not None:
            if self.breaker_policy.failure_threshold <= 0 or self.breaker_policy.cooldown_rounds <= 0:
                raise ValueError("breaker policy values must be positive")
        if self.cost_class == "H" and self.breaker_policy is not None and self.fallback_route is None:
            raise ValueError("high-cost breaker configuration requires a fallback route")


@dataclass
class AgentSpec:
    name: str
    class_kind: str
    wakeup_rule: str
    owned_skills: list[str]
    budget_class: str
    fallback_policy: str
    config_key: str | None = None


@dataclass
class SkillInvocation:
    round_id: int
    skill_name: str
    owner_module: str
    inputs: dict[str, Any]
    seed_ref: int | None = None


@dataclass
class SkillResult:
    skill_name: str
    owner_module: str
    output: dict[str, Any]
    latency_ms: int
    cost_class: str
    input_hash: str = ""
    output_hash: str = ""
    degraded: bool = False
    failure_policy_applied: str | None = None
    seed_ref: int | None = None
    fallback_route: str | None = None
    fallback_cost_class: str | None = None
    policy_rejection_reason: str | None = None
    breaker_state: dict[str, Any] = field(default_factory=dict)
    parallel_group: str | None = None
    agent_tier: str | None = None
    task_priority: str | None = None
    task_outcome: str | None = None
    task_type: str | None = None
    timeout_ms: int | None = None


@dataclass
class ActionEvidenceSignal:
    module_name: str
    module_type: str
    confidence: float = 0.5
    action_delta: dict[str, float] = field(default_factory=dict)
    utility_shift: dict[str, float] = field(default_factory=dict)
    sigma_scale: float = 1.0
    priority_bucket: str = "task_goal"
    control_domain: str = "task"
    gated_actions: list[str] = field(default_factory=list)
    risk_hints: dict[str, Any] = field(default_factory=dict)
    veto: bool = False
    trace_reason: str = ""
    trace_tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.confidence = round(max(0.0, min(1.0, float(self.confidence))), 4)
        self.sigma_scale = round(max(0.35, min(1.6, float(self.sigma_scale))), 4)


@dataclass
class EnergyProjectionSpec:
    module_type: str
    target_space: Literal["context", "memory", "action", "token", "global"]
    normalization_strategy: str = "rms"
    module_temperature: float = 1.0
    variance_clip: float = 3.0
    coupling_mode: str = "layer_local"


@dataclass
class DeltaStatistics:
    support_size: int = 0
    mean_abs: float = 0.0
    rms: float = 0.0
    max_abs: float = 0.0
    clipped: bool = False


@dataclass
class CrossLayerCouplingSpec:
    source_layer: Literal["context", "memory", "action", "token", "global"]
    target_layer: Literal["context", "memory", "action", "token", "global"]
    carrier_signal: str
    projection_rule: str
    allowed_phase: str
    enabled: bool = True


@dataclass
class ProbabilisticContribution:
    module_name: str
    module_type: str
    level: Literal["context", "memory", "action", "token", "global"]
    target_space: Literal["context", "memory", "action", "token", "global"]
    raw_signal: dict[str, float] = field(default_factory=dict)
    modulated_delta: dict[str, float] = field(default_factory=dict)
    inhibitory_drive: dict[str, float] = field(default_factory=dict)
    failure_taxonomy: list[str] = field(default_factory=list)
    hard_mask: dict[str, bool] = field(default_factory=dict)
    posterior: dict[str, float] = field(default_factory=dict)
    peak_clusters: list[dict[str, Any]] = field(default_factory=list)
    compromise_template_prior: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5
    confidence_calibrated: float | None = None
    trace_reason: str = ""
    projection_reason: str = ""
    applied_at_stage: str = ""
    native_operator: str = "modulated_delta"
    dependency_trace: list[str] = field(default_factory=list)
    projection: EnergyProjectionSpec | None = None

    def __post_init__(self) -> None:
        self.raw_signal = _normalize_signal_map(self.raw_signal)
        self.modulated_delta = _normalize_signal_map(self.modulated_delta)
        self.inhibitory_drive = _normalize_positive_signal_map(self.inhibitory_drive)
        self.failure_taxonomy = _normalize_string_list(self.failure_taxonomy)
        self.hard_mask = {
            str(target).strip(): bool(flag)
            for target, flag in dict(self.hard_mask or {}).items()
            if str(target).strip()
        }
        if not self.raw_signal:
            self.raw_signal = dict(self.modulated_delta)
        if not self.modulated_delta:
            self.modulated_delta = dict(self.raw_signal)
        self.confidence = round(max(0.0, min(1.0, float(self.confidence))), 4)
        if self.confidence_calibrated is not None:
            self.confidence_calibrated = round(max(0.0, min(1.0, float(self.confidence_calibrated))), 4)
        if self.projection is None:
            self.projection = EnergyProjectionSpec(module_type=self.module_type, target_space=self.target_space)


@dataclass
class ContributionAuditRecord:
    module_name: str
    module_type: str
    level: Literal["context", "memory", "action", "token", "global"]
    target_space: Literal["context", "memory", "action", "token", "global"]
    delta_raw: dict[str, float] = field(default_factory=dict)
    delta_projected: dict[str, float] = field(default_factory=dict)
    delta_normalized: dict[str, float] = field(default_factory=dict)
    raw_signal: dict[str, float] = field(default_factory=dict)
    modulated_delta: dict[str, float] = field(default_factory=dict)
    inhibitory_drive: dict[str, float] = field(default_factory=dict)
    failure_taxonomy: list[str] = field(default_factory=list)
    hard_masked_targets: list[str] = field(default_factory=list)
    posterior: dict[str, float] = field(default_factory=dict)
    peak_clusters: list[dict[str, Any]] = field(default_factory=list)
    compromise_template_prior: dict[str, float] = field(default_factory=dict)
    confidence_raw: float = 0.0
    confidence_calibrated: float = 0.0
    normalization_reason: str = ""
    projection_reason: str = ""
    trace_reason: str = ""
    dependency_trace: list[str] = field(default_factory=list)
    stats: DeltaStatistics = field(default_factory=DeltaStatistics)


@dataclass
class ProbabilityLayerState:
    layer: Literal["context", "memory", "action", "token", "global"]
    base_energy: dict[str, float] = field(default_factory=dict)
    aggregated_raw_signal: dict[str, float] = field(default_factory=dict)
    aggregated_modulated_delta: dict[str, float] = field(default_factory=dict)
    aggregated_inhibitory_drive: dict[str, float] = field(default_factory=dict)
    aggregated_projected_delta: dict[str, float] = field(default_factory=dict)
    aggregated_normalized_delta: dict[str, float] = field(default_factory=dict)
    final_energy: dict[str, float] = field(default_factory=dict)
    failure_taxonomy: list[str] = field(default_factory=list)
    hard_masked_targets: list[str] = field(default_factory=list)
    winner_target: str = ""
    winner_posterior: dict[str, float] = field(default_factory=dict)
    counterfactual_top_peaks: list[dict[str, Any]] = field(default_factory=list)
    contribution_audit: list[ContributionAuditRecord] = field(default_factory=list)


@dataclass
class ProbabilityFieldSnapshot:
    context: ProbabilityLayerState = field(default_factory=lambda: ProbabilityLayerState(layer="context"))
    memory: ProbabilityLayerState = field(default_factory=lambda: ProbabilityLayerState(layer="memory"))
    action: ProbabilityLayerState = field(default_factory=lambda: ProbabilityLayerState(layer="action"))
    token: ProbabilityLayerState = field(default_factory=lambda: ProbabilityLayerState(layer="token"))
    token_state: TokenFieldState | None = None
    couplings: list[CrossLayerCouplingSpec] = field(default_factory=list)
    source_chain: list[str] = field(default_factory=list)


@dataclass
class TokenFieldState:
    step_index: int
    prefix_tokens: list[str] = field(default_factory=list)
    active_module_sources: list[str] = field(default_factory=list)
    generated_delta_sources: list[str] = field(default_factory=list)
    delta_generation_policy: Literal["propagate_only"] = "propagate_only"
    sequence_consistency_tag: str = "persistent_field"

    def __post_init__(self) -> None:
        illegal = [source for source in self.generated_delta_sources if source not in self.active_module_sources]
        if illegal:
            joined = ", ".join(illegal)
            raise ValueError(f"TokenFieldState cannot generate new delta sources without explicit module registration: {joined}")


@dataclass
class GateDecision:
    owner: str
    allowed: bool
    reason: str
    action_overrides: dict[str, float] = field(default_factory=dict)
    requires_resample: bool = False
    forced_mode: str | None = None


@dataclass
class RoundContext:
    round_id: int
    scenario: str
    mode: str
    thresholds: dict[str, Any]
    budgets: dict[str, Any]
    scenario_config: dict[str, Any]
    seed_ref: int | None = None
    wake_flags: dict[str, bool] = field(default_factory=dict)
    policy_flags: dict[str, Any] = field(default_factory=dict)


@dataclass
class CommandEnvelope:
    command_id: str
    domain: str
    verb: str
    target: str | None = None
    canonical: str = ""
    mutation_scope: str | None = None
    parsed_args: dict[str, Any] = field(default_factory=dict)
    flags: dict[str, Any] = field(default_factory=dict)
    operator_level: str = "read_only"
    ttl: str | None = None
    rollback_available: bool = False


@dataclass
class ActionBookkeepingState:
    u_base: dict[str, float] = field(default_factory=dict)
    p_base: dict[str, float] = field(default_factory=dict)
    ci: dict[str, float] = field(default_factory=dict)
    gate: dict[str, float] = field(default_factory=dict)
    risk_suppressor: dict[str, float] = field(default_factory=dict)
    query_intent: dict[str, Any] = field(default_factory=dict)
    disclosure_intent: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictComponents:
    proposal_divergence: float = 0.0
    veto_tension: float = 0.0
    value_gap: float = 0.0
    relation_risk_gap: float = 0.0
    body_gap: float = 0.0


@dataclass
class ConflictAssessment:
    score: float = 0.0
    components: ConflictComponents = field(default_factory=ConflictComponents)
    priority_signals: dict[str, float] = field(default_factory=dict)
    dominant_conflicts: list[str] = field(default_factory=list)
    critical_conflict: bool = False


@dataclass
class ConflictResolution:
    flag: bool = False
    winning_priority: str | None = None
    blocked_actions: list[str] = field(default_factory=list)
    action_scales: dict[str, float] = field(default_factory=dict)
    applied_template: str | None = None
    reason: str = ""
    post_error_adjustment: ConflictPostErrorAdjustment = field(default_factory=ConflictPostErrorAdjustment)
    repair_transition: dict[str, Any] = field(default_factory=dict)
    repair_state_snapshot: ConflictRepairState = field(default_factory=ConflictRepairState)
    repair_ledger_tail: list[ConflictRepairLedgerEntry] = field(default_factory=list)


@dataclass
class ConflictPassRecord:
    pass_index: int
    assessment: ConflictAssessment
    resolution: ConflictResolution
    resample_requested: bool = False


@dataclass
class CompromiseDecision:
    triggered: bool = False
    template: str | None = None
    reason: str = ""
    winning_priority: str | None = None


@dataclass
class ConflictCircuitState:
    triggered: bool = False
    active: bool = False
    hot_rounds_remaining: int = 0
    recovery_rounds_remaining: int = 0
    blocked_actions: list[str] = field(default_factory=list)


@dataclass
class IntentPosterior:
    top_intent: str = ""
    posterior: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    clipped: list[str] = field(default_factory=list)


@dataclass
class QueryIntentState:
    posterior: IntentPosterior = field(default_factory=IntentPosterior)
    legacy_query_kind: str = "general"


@dataclass
class DisclosureIntentState:
    posterior: IntentPosterior = field(default_factory=IntentPosterior)
    legacy_disclosure_detail: str = "none"


@dataclass
class IntentTraceRecord:
    query: QueryIntentState = field(default_factory=QueryIntentState)
    disclosure: DisclosureIntentState = field(default_factory=DisclosureIntentState)


@dataclass
class QuantumEntropyRef:
    source: str = ""
    fetched_at: str = ""
    batch_id: str = ""
    byte_start: int = 0
    byte_length: int = 0
    purpose: str = ""
    node_name: str = ""
    provider_class: str = ""
    endpoint: str = ""
    health_state: str = "ready"
    hard_block_triggered: bool = False
    failure_class: str = ""
    degraded: bool = False
    reason: str = ""


@dataclass
class QuantumEntropyBatch:
    source: str = ""
    fetched_at: str = ""
    batch_id: str = ""
    total_bytes: int = 0
    available_bytes: int = 0
    provider_class: str = ""
    endpoint: str = ""
    health_state: str = "ready"
    failure_class: str = ""
    degraded: bool = False
    reason: str = ""


@dataclass
class StochasticState:
    emo_channel: str
    xi_emo: float
    xi_mood: float
    lambda_noise: float
    r_intensity: float
    noise_guard_triggered: bool
    round_seed: int
    kl_divergence: float = 0.0
    timing_jitter: float = 0.0
    fragmentation_jitter: float = 0.0
    v_t: float = 0.0
    v_t_components: dict[str, float] = field(default_factory=dict)
    sigma_emo: float = 0.0
    sigma_mood: float = 0.0
    lambda_noise_pre_guard: float = 0.0
    log_m_guard_triggered: bool = False
    guard_reason: str = ""
    base_stochastic_distribution: dict[str, float] = field(default_factory=dict)
    q_noise_distribution: dict[str, float] = field(default_factory=dict)
    q_noise_pre_guard_summary: dict[str, float] = field(default_factory=dict)
    winner_flip_detected: bool = False
    winner_flip_from: str = ""
    winner_flip_to: str = ""
    entropy_refs_by_node: dict[str, Any] = field(default_factory=dict)
    entropy_ref: QuantumEntropyRef = field(default_factory=QuantumEntropyRef)


@dataclass
class ExpressionProfile:
    reply_delay: float
    latency_style: float
    sentence_fragmentation: float
    hedging_level: float
    warmth_level: float
    directness_level: float
    self_disclosure: float
    tone_sharpness: float
    repair_tendency: float
    timing_jitter: float
    fragmentation_jitter: float


@dataclass
class RenderPlan:
    action: str
    expression: ExpressionProfile
    delivery_mode: Literal["speech", "monologue"] = "speech"
    safety_constraints: dict[str, Any] = field(default_factory=dict)
    message_plan: dict[str, Any] = field(default_factory=dict)
    event_summary: str = ""
    scenario: str = ""
    target: str | None = None
    relation_state: dict[str, Any] = field(default_factory=dict)
    perspective: dict[str, Any] = field(default_factory=dict)
    identity_context: IdentityContext = field(default_factory=IdentityContext)

    def __post_init__(self) -> None:
        if isinstance(self.expression, dict):
            self.expression = ExpressionProfile(**self.expression)
        self.delivery_mode = "monologue" if str(self.delivery_mode or "speech") == "monologue" else "speech"
        if isinstance(self.identity_context, dict):
            self.identity_context = IdentityContext(**self.identity_context)


@dataclass
class RenderedExpression:
    text: str
    route: str
    model: str
    delivery_mode: Literal["speech", "monologue"] = "speech"
    degraded: bool = False
    failure_policy_applied: str | None = None
    authenticity: AuthenticityRecord = field(default_factory=AuthenticityRecord)

    def __post_init__(self) -> None:
        self.delivery_mode = "monologue" if str(self.delivery_mode or "speech") == "monologue" else "speech"
        if isinstance(self.authenticity, dict):
            self.authenticity = AuthenticityRecord(**self.authenticity)
