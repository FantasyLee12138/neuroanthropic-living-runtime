from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
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
    round_id: int
    scenario: str
    mode: str
    sampled_action: str
    contributions: list[AgentContribution]
    top_drivers: list[AgentContribution]
    style_profile: dict[str, Any]
    state_snapshot: dict[str, Any]


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
    budget_remaining: float = 1.0
    last_action: str = "boot"
    last_checkpoint_id: str | None = None
    agents_enabled: dict[str, bool] = field(default_factory=dict)


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

