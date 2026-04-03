from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return to_dict(asdict(value))
    if isinstance(value, list):
        return [to_dict(item) for item in value]
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
    provider: str = "rule"
    model: str = "fallback"
    latency_ms: int = 0


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
    delta_p: float = 0.0
    sigma_scale: float = 1.0
    confidence: float = 0.0
    weight_applied: float = 1.0
    resample_idx: int = 0
    selected: bool = False
    latency_ms: int = 0
    provider: str = "rule"
    model: str = "fallback"
    tags: list[str] = field(default_factory=list)


@dataclass
class RoundTrace:
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
    resample_count: int = 0
    pre_state_snapshot: dict[str, Any] = field(default_factory=dict)
    event_payload: dict[str, Any] = field(default_factory=dict)
    decision_context: dict[str, Any] = field(default_factory=dict)
    candidate_distribution: dict[str, float] = field(default_factory=dict)
    conflict_score: float = 0.0
    plausibility_fail_score: float = 0.0
    rendered_output: str = ""
    provider: str = "rule_fallback"
    model: str = "fallback"


@dataclass
class HealthEvent:
    event: str
    status: str
    detail: str


@dataclass
class RuntimeState:
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
    last_render_provider: str = "rule_fallback"
    last_render_model: str = "fallback"
    last_conflict_score: float = 0.0
    last_plausibility_fail_score: float = 0.0


@dataclass
class RoundResult:
    round_id: int
    sampled_action: ActionCandidate
    trace: RoundTrace
    state: RuntimeState
    health: HealthEvent


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


@dataclass
class SkillSpec:
    name: str
    owner_module: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    timeout_ms: int
    cost_class: str
    failure_policy: str
    trace_tags: list[str]
    skill_kind: str = "scoring"
    sync_mode: str = "sync"
    policy_check: bool = False
    idempotent: bool = True
    output_kind: str = "scalar"


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
    p_base: dict[str, float] = field(default_factory=dict)
    p_raw: dict[str, float] = field(default_factory=dict)
    p_mix: dict[str, float] = field(default_factory=dict)
    p_final: dict[str, float] = field(default_factory=dict)
    ci: dict[str, float] = field(default_factory=dict)
    gate: dict[str, float] = field(default_factory=dict)
    risk_suppressor: dict[str, float] = field(default_factory=dict)
    resample_idx: int = 0


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
