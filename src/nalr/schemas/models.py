from __future__ import annotations

from dataclasses import asdict, dataclass, field
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


@dataclass
class Proposal:
    agent_name: str
    action_preferences: dict[str, float]
    confidence: float
    sigma_scale: float = 1.0
    veto: bool = False
    trace_tags: list[str] = field(default_factory=list)
    reason: str = ""


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
    endogenous_tick_reason: dict[str, Any] = field(default_factory=dict)
    endogenous_policy_shift: dict[str, Any] = field(default_factory=dict)
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
    model_call_traces: list[dict[str, Any]] = field(default_factory=list)
    runtime_metrics: dict[str, Any] = field(default_factory=dict)
    resample_count: int = 0


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
class TurnPlan:
    text: str
    route: str
    scenario: str
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
    assistant_preamble: str = ""
    assistant_final: str = ""
    run: dict[str, Any] = field(default_factory=dict)
    explain: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


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
class RuntimeState:
    session_id: str = field(default_factory=lambda: uuid4().hex)
    subject_core: SubjectCore = field(default_factory=SubjectCore)
    mode: str = "interactive"
    safe_mode: bool = False
    round_count: int = 0
    body_energy: float = 0.7
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
    motivation_pool_state: MotivationPoolState = field(default_factory=MotivationPoolState)
    motivation_learning_state: MotivationLearningState = field(default_factory=MotivationLearningState)
    endogenous_scheduler_state: EndogenousSchedulerState = field(default_factory=EndogenousSchedulerState)
    endogenous_state: dict[str, Any] = field(default_factory=dict)
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
        if isinstance(self.subject_core, dict):
            self.subject_core = SubjectCore(**self.subject_core)
        if isinstance(self.identity_state, dict):
            self.identity_state = IdentityState(**self.identity_state)
        if isinstance(self.motivation_pool_state, dict):
            self.motivation_pool_state = MotivationPoolState(**self.motivation_pool_state)
        if isinstance(self.motivation_learning_state, dict):
            self.motivation_learning_state = MotivationLearningState(**self.motivation_learning_state)
        if isinstance(self.endogenous_scheduler_state, dict):
            self.endogenous_scheduler_state = EndogenousSchedulerState(**self.endogenous_scheduler_state)
        self.identity_state.aliases = sanitize_identity_aliases(self.identity_state.aliases)
        self.temperament_state = normalize_temperament_state(self.temperament_state)
        if isinstance(self.repair_state, dict):
            self.repair_state = ConflictRepairState(**self.repair_state)
        if isinstance(self.last_post_error_adjustment, dict):
            self.last_post_error_adjustment = ConflictPostErrorAdjustment(**self.last_post_error_adjustment)
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
        if not isinstance(self.endogenous_state, dict):
            self.endogenous_state = {}
        self.endogenous_state.setdefault("current_intent", None)
        self.endogenous_state.setdefault("stability", 0)
        self.endogenous_state.setdefault("history", [])
        self.endogenous_state.setdefault("last_trigger", "")


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
        if isinstance(self.identity_context, dict):
            self.identity_context = IdentityContext(**self.identity_context)


@dataclass
class RenderedExpression:
    text: str
    route: str
    model: str
    degraded: bool = False
    failure_policy_applied: str | None = None
    authenticity: AuthenticityRecord = field(default_factory=AuthenticityRecord)

    def __post_init__(self) -> None:
        if isinstance(self.authenticity, dict):
            self.authenticity = AuthenticityRecord(**self.authenticity)
