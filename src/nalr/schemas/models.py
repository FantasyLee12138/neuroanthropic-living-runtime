from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
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


@dataclass
class RoundEvent:
    source: str
    content: str
    target: str | None = None
    cue: str | None = None
    valence: float = 0.0
    energy_delta: float = 0.0


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
    scenario: str
    mode: str
    sampled_action: str
    contributions: list[AgentContribution]
    top_drivers: list[AgentContribution]
    style_profile: dict[str, Any]
    state_snapshot: dict[str, Any]
    pipeline_stages: list[str] = field(default_factory=list)
    proposal_summaries: list[dict[str, Any]] = field(default_factory=list)
    gate_decisions: list[dict[str, Any]] = field(default_factory=list)
    skill_traces: list[dict[str, Any]] = field(default_factory=list)
    distribution_state: dict[str, Any] = field(default_factory=dict)
    stochastic_state: dict[str, Any] = field(default_factory=dict)
    render_plan: dict[str, Any] = field(default_factory=dict)
    rendered_expression: dict[str, Any] = field(default_factory=dict)
    resample_count: int = 0


@dataclass
class HealthEvent:
    event: str
    status: str
    detail: str


@dataclass
class RuntimeState:
    session_id: str = field(default_factory=lambda: uuid4().hex)
    mode: str = "interactive"
    safe_mode: bool = False
    round_count: int = 0
    body_energy: float = 0.7
    mood: float = 0.55
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
    temperament_state: dict[str, float] = field(default_factory=dict)
    resource_state: dict[str, float] = field(default_factory=dict)
    repair_mode: str | None = None
    session_metadata: dict[str, Any] = field(default_factory=dict)


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


@dataclass
class ProposalBundle:
    owner: str
    confidence: float = 0.5
    action_preferences: dict[str, float] = field(default_factory=dict)
    delta_p: dict[str, float] = field(default_factory=dict)
    sigma_scale: float = 1.0
    veto: bool = False
    mode_switch: str | None = None
    utility_shift: dict[str, float] = field(default_factory=dict)
    state_patch: dict[str, Any] = field(default_factory=dict)
    memory_ops: list[dict[str, Any]] = field(default_factory=list)
    trace_tags: list[str] = field(default_factory=list)
    reason: str = ""
    priority_bucket: str = "task_goal"
    control_domain: str = "task"
    gated_actions: list[str] = field(default_factory=list)
    risk_hints: dict[str, Any] = field(default_factory=dict)


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
    domain: str
    verb: str
    target: str | None = None
    flags: dict[str, Any] = field(default_factory=dict)
    operator_level: str = "read_only"
    ttl: str | None = None
    rollback_available: bool = False


@dataclass
class ActionDistributionState:
    u_base: dict[str, float] = field(default_factory=dict)
    u_shifted: dict[str, float] = field(default_factory=dict)
    p_base: dict[str, float] = field(default_factory=dict)
    p_raw: dict[str, float] = field(default_factory=dict)
    p_base_stochastic: dict[str, float] = field(default_factory=dict)
    q_noise: dict[str, float] = field(default_factory=dict)
    p_mix: dict[str, float] = field(default_factory=dict)
    p_final: dict[str, float] = field(default_factory=dict)
    ci: dict[str, float] = field(default_factory=dict)
    gate: dict[str, float] = field(default_factory=dict)
    risk_suppressor: dict[str, float] = field(default_factory=dict)
    resample_idx: int = 0
    conflict_mode: str = "none"
    conflict: dict[str, Any] = field(default_factory=dict)


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
class StochasticState:
    emo_channel: str
    xi_emo: float
    xi_mood: float
    lambda_noise: float
    r_intensity: float
    noise_guard_triggered: bool
    round_seed: int
    kl_divergence: float = 0.0


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


@dataclass
class RenderedExpression:
    text: str
    route: str
    model: str
    degraded: bool = False
    failure_policy_applied: str | None = None
