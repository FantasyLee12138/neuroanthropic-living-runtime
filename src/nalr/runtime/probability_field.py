from __future__ import annotations

import math
from typing import Iterable

from nalr.schemas.models import (
    ContributionAuditRecord,
    CrossLayerCouplingSpec,
    DeltaStatistics,
    EnergyProjectionSpec,
    ProbabilityFieldSnapshot,
    ProbabilityLayerState,
    ProbabilisticContribution,
    TokenFieldState,
)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _stats(delta: dict[str, float]) -> DeltaStatistics:
    if not delta:
        return DeltaStatistics()
    values = [float(value) for value in delta.values()]
    mean_abs = sum(abs(value) for value in values) / len(values)
    rms = math.sqrt(sum(value * value for value in values) / len(values))
    return DeltaStatistics(
        support_size=len(values),
        mean_abs=round(mean_abs, 6),
        rms=round(rms, 6),
        max_abs=round(max(abs(value) for value in values), 6),
        clipped=False,
    )


def _merge_delta_maps(*payloads: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for payload in payloads:
        for target, value in dict(payload or {}).items():
            key = str(target).strip()
            if not key:
                continue
            merged[key] = round(merged.get(key, 0.0) + float(value), 6)
    return merged


def _posterior_from_energy(energy: dict[str, float]) -> dict[str, float]:
    finite = {str(target): float(value) for target, value in energy.items() if float(value) != float("-inf")}
    if not finite:
        return {}
    anchor = max(finite.values())
    exp_terms = {target: math.exp(value - anchor) for target, value in finite.items()}
    normalizer = sum(exp_terms.values()) or 1.0
    return {target: round(weight / normalizer, 6) for target, weight in exp_terms.items()}


def _top_peaks(
    energy: dict[str, float],
    posterior: dict[str, float],
    hard_masked: set[str],
    *,
    limit: int = 3,
) -> list[dict[str, float | str | bool]]:
    rows: list[dict[str, float | str | bool]] = []
    for target, value in energy.items():
        if float(value) == float("-inf"):
            continue
        rows.append(
            {
                "target": str(target),
                "final_energy": round(float(value), 6),
                "posterior": round(float(posterior.get(str(target), 0.0) or 0.0), 6),
                "hard_masked": str(target) in hard_masked,
            }
        )
    rows.sort(key=lambda item: (float(item["posterior"]), float(item["final_energy"])), reverse=True)
    return rows[:limit]


class DeltaNormalizationLayer:
    def normalize(self, delta: dict[str, float], spec: EnergyProjectionSpec) -> tuple[dict[str, float], DeltaStatistics]:
        stats = _stats(delta)
        if not delta:
            return {}, stats

        base_scale = max(stats.rms, 0.35)
        temperature = max(0.1, float(spec.module_temperature or 1.0))
        clip_limit = max(0.25, float(spec.variance_clip or 3.0))
        clipped = False
        normalized: dict[str, float] = {}
        for target, raw in delta.items():
            value = (float(raw) / base_scale) / temperature
            bounded = _clip(value, -clip_limit, clip_limit)
            if bounded != value:
                clipped = True
            normalized[target] = round(bounded, 6)
        stats.clipped = clipped
        return normalized, stats


class ProbabilityFieldIntegrator:
    ALLOWED_COUPLINGS = {
        ("context", "memory", "context_route"),
        ("memory", "action", "memory_prior"),
        ("action", "token", "render_plan"),
    }

    def __init__(self) -> None:
        self.normalizer = DeltaNormalizationLayer()

    def integrate(
        self,
        *,
        context_base: dict[str, float] | None = None,
        memory_base: dict[str, float] | None = None,
        action_base: dict[str, float] | None = None,
        token_base: dict[str, float] | None = None,
        contributions: Iterable[ProbabilisticContribution],
        token_state: TokenFieldState | None = None,
        couplings: list[CrossLayerCouplingSpec] | None = None,
        source_chain: list[str] | None = None,
    ) -> ProbabilityFieldSnapshot:
        contribution_rows = list(contributions)
        coupling_rows = list(couplings or [])
        self._validate_couplings(coupling_rows)
        self._validate_token_sources(contribution_rows, token_state)
        context_state = self._integrate_layer("context", context_base or {}, contribution_rows)
        memory_state = self._integrate_layer(
            "memory",
            self._propagate_base_energy(
                target_layer="memory",
                base_energy=memory_base or {},
                source_states={"context": context_state},
                couplings=coupling_rows,
            ),
            contribution_rows,
        )
        action_state = self._integrate_layer(
            "action",
            self._propagate_base_energy(
                target_layer="action",
                base_energy=action_base or {},
                source_states={"memory": memory_state},
                couplings=coupling_rows,
            ),
            contribution_rows,
        )
        token_layer = self._integrate_layer(
            "token",
            self._propagate_base_energy(
                target_layer="token",
                base_energy=token_base or {},
                source_states={"action": action_state},
                couplings=coupling_rows,
            ),
            contribution_rows,
        )
        grouped = {
            "context": context_state,
            "memory": memory_state,
            "action": action_state,
            "token": token_layer,
        }
        return self._compose_snapshot(
            context=grouped["context"],
            memory=grouped["memory"],
            action=grouped["action"],
            token=grouped["token"],
            token_state=token_state,
            couplings=coupling_rows,
            source_chain=source_chain,
        )

    def reintegrate_action_token_layers(
        self,
        *,
        snapshot: ProbabilityFieldSnapshot,
        contributions: Iterable[ProbabilisticContribution],
        action_base: dict[str, float] | None = None,
        token_base: dict[str, float] | None = None,
        token_state: TokenFieldState | None = None,
        couplings: list[CrossLayerCouplingSpec] | None = None,
        source_chain: list[str] | None = None,
    ) -> ProbabilityFieldSnapshot:
        contribution_rows = list(contributions)
        coupling_rows = list(snapshot.couplings if couplings is None else couplings)
        effective_token_state = snapshot.token_state if token_state is None else token_state
        self._validate_couplings(coupling_rows)
        self._validate_token_sources(contribution_rows, effective_token_state)
        action_state = self._integrate_layer(
            "action",
            self._propagate_base_energy(
                target_layer="action",
                base_energy=snapshot.action.base_energy if action_base is None else action_base,
                source_states={"memory": snapshot.memory},
                couplings=coupling_rows,
            ),
            contribution_rows,
        )
        token_layer = self._integrate_layer(
            "token",
            self._propagate_base_energy(
                target_layer="token",
                base_energy=snapshot.token.base_energy if token_base is None else token_base,
                source_states={"action": action_state},
                couplings=coupling_rows,
            ),
            contribution_rows,
        )
        return self._compose_snapshot(
            context=snapshot.context,
            memory=snapshot.memory,
            action=action_state,
            token=token_layer,
            token_state=effective_token_state,
            couplings=coupling_rows,
            source_chain=source_chain,
            inherited_source_chain=snapshot.source_chain,
        )

    def _compose_snapshot(
        self,
        *,
        context: ProbabilityLayerState,
        memory: ProbabilityLayerState,
        action: ProbabilityLayerState,
        token: ProbabilityLayerState,
        token_state: TokenFieldState | None,
        couplings: list[CrossLayerCouplingSpec],
        source_chain: list[str] | None = None,
        inherited_source_chain: Iterable[str] | None = None,
    ) -> ProbabilityFieldSnapshot:
        merged_source_chain = self._merge_source_chain(
            inherited_source_chain=inherited_source_chain,
            source_chain=source_chain,
        )
        return ProbabilityFieldSnapshot(
            context=context,
            memory=memory,
            action=action,
            token=token,
            token_state=token_state,
            couplings=couplings,
            source_chain=merged_source_chain,
        )

    def _merge_source_chain(
        self,
        *,
        inherited_source_chain: Iterable[str] | None,
        source_chain: list[str] | None,
    ) -> list[str]:
        merged: list[str] = []
        for item in list(inherited_source_chain or []) + list(source_chain or []):
            entry = str(item).strip()
            if not entry:
                continue
            if entry not in merged:
                merged.append(entry)
        return merged

    def _propagate_base_energy(
        self,
        *,
        target_layer: str,
        base_energy: dict[str, float],
        source_states: dict[str, ProbabilityLayerState],
        couplings: list[CrossLayerCouplingSpec],
    ) -> dict[str, float]:
        propagated = {str(key): round(float(value), 6) for key, value in dict(base_energy).items()}
        for coupling in couplings:
            if coupling.target_layer != target_layer or not coupling.enabled:
                continue
            if (
                coupling.source_layer == "context"
                and coupling.target_layer == "memory"
                and coupling.carrier_signal == "context_route"
            ):
                context_state = source_states.get("context")
                if context_state is None:
                    continue
                for key, value in context_state.final_energy.items():
                    key_name = str(key)
                    if not key_name.startswith("cue:"):
                        continue
                    cue = key_name.split(":", 1)[1].strip()
                    if not cue:
                        continue
                    propagated[cue] = round(propagated.get(cue, 0.0) + max(0.0, float(value)) * 0.25, 6)
            elif (
                coupling.source_layer == "memory"
                and coupling.target_layer == "action"
                and coupling.carrier_signal == "memory_prior"
            ):
                memory_state = source_states.get("memory")
                if memory_state is None:
                    continue
                cue_strengths = [
                    max(0.0, float(value))
                    for key, value in memory_state.final_energy.items()
                    if ":" not in str(key) and str(key).strip()
                ]
                if not cue_strengths:
                    continue
                propagated["recall"] = round(propagated.get("recall", 0.0) + max(cue_strengths) * 0.2, 6)
            elif (
                coupling.source_layer == "action"
                and coupling.target_layer == "token"
                and coupling.carrier_signal == "render_plan"
            ):
                action_state = source_states.get("action")
                if action_state is None:
                    continue
                posterior = dict(action_state.winner_posterior or {}) or _posterior_from_energy(action_state.final_energy)
                for action, value in posterior.items():
                    action_name = str(action).strip()
                    if not action_name or ":" in action_name:
                        continue
                    propagated[f"act:{action_name}"] = round(
                        propagated.get(f"act:{action_name}", 0.0) + max(0.0, float(value)) * 0.2,
                        6,
                    )
        return propagated

    def _validate_couplings(self, couplings: list[CrossLayerCouplingSpec]) -> None:
        for coupling in couplings:
            pair = (coupling.source_layer, coupling.target_layer, coupling.carrier_signal)
            if pair not in self.ALLOWED_COUPLINGS:
                raise ValueError(
                    f"Illegal cross-layer coupling: {coupling.source_layer}->{coupling.target_layer}:{coupling.carrier_signal}"
                )
            if not coupling.enabled:
                raise ValueError(
                    f"Disabled cross-layer coupling cannot enter probability field: {coupling.source_layer}->{coupling.target_layer}:{coupling.carrier_signal}"
                )

    def _validate_token_sources(
        self,
        contributions: list[ProbabilisticContribution],
        token_state: TokenFieldState | None,
    ) -> None:
        token_rows = [row for row in contributions if row.level == "token" or row.target_space == "token"]
        if not token_rows:
            return
        if token_state is None:
            raise ValueError("Token contributions require TokenFieldState")
        allowed = set(token_state.active_module_sources)
        illegal = sorted({row.module_name for row in token_rows if row.module_name not in allowed})
        if illegal:
            raise ValueError(
                f"Token contributions must come from declared active module sources: {', '.join(illegal)}"
            )

    def _layer_accepts_key(self, layer: str, key: str) -> bool:
        if layer == "action":
            return ":" not in key
        if layer == "token":
            return ":" in key
        return True

    def _filter_layer_energy(self, layer: str, energy: dict[str, float]) -> dict[str, float]:
        return {str(key): round(float(value), 6) for key, value in dict(energy).items() if self._layer_accepts_key(layer, str(key))}

    def _integrate_layer(
        self,
        layer: str,
        base_energy: dict[str, float],
        contributions: list[ProbabilisticContribution],
    ) -> ProbabilityLayerState:
        relevant = [item for item in contributions if item.target_space == layer or item.level == layer]
        base_energy = self._filter_layer_energy(layer, base_energy)
        if layer == "token":
            self._validate_token_action_alignment(base_energy, relevant)
        aggregated_raw_signal: dict[str, float] = {}
        aggregated_modulated_delta: dict[str, float] = {}
        aggregated_inhibitory_drive: dict[str, float] = {}
        aggregated_projected: dict[str, float] = {}
        aggregated_normalized: dict[str, float] = {}
        failure_taxonomy: list[str] = []
        hard_masked: set[str] = set()
        audits: list[ContributionAuditRecord] = []

        for contribution in relevant:
            raw_signal = self._filter_layer_energy(layer, self._raw_signal(contribution))
            modulated_delta = self._filter_layer_energy(layer, self._modulated_delta(contribution))
            inhibitory_drive = self._filter_layer_energy(layer, self._inhibitory_drive(contribution))
            projected = self._filter_layer_energy(layer, self._project(contribution))
            normalized, stats = self.normalizer.normalize(projected, contribution.projection or EnergyProjectionSpec(module_type=contribution.module_type, target_space=layer))
            for target, value in raw_signal.items():
                aggregated_raw_signal[target] = round(aggregated_raw_signal.get(target, 0.0) + float(value), 6)
            for target, value in modulated_delta.items():
                aggregated_modulated_delta[target] = round(aggregated_modulated_delta.get(target, 0.0) + float(value), 6)
            for target, value in inhibitory_drive.items():
                aggregated_inhibitory_drive[target] = round(aggregated_inhibitory_drive.get(target, 0.0) + float(value), 6)
            for target, value in projected.items():
                aggregated_projected[target] = round(aggregated_projected.get(target, 0.0) + float(value), 6)
            for target, value in normalized.items():
                calibrated = contribution.confidence_calibrated if contribution.confidence_calibrated is not None else contribution.confidence
                weighted = float(value) * max(0.0, min(1.0, float(calibrated)))
                aggregated_normalized[target] = round(aggregated_normalized.get(target, 0.0) + weighted, 6)
            for item in contribution.failure_taxonomy:
                if item not in failure_taxonomy:
                    failure_taxonomy.append(item)
            hard_masked.update(target for target, flag in contribution.hard_mask.items() if flag and self._layer_accepts_key(layer, str(target)))
            audits.append(
                ContributionAuditRecord(
                    module_name=contribution.module_name,
                    module_type=contribution.module_type,
                    level=contribution.level,
                    target_space=contribution.target_space,
                    delta_raw=raw_signal,
                    delta_projected=projected,
                    delta_normalized=normalized,
                    raw_signal=raw_signal,
                    modulated_delta=modulated_delta,
                    inhibitory_drive=inhibitory_drive,
                    failure_taxonomy=list(contribution.failure_taxonomy),
                    hard_masked_targets=[
                        target for target, flag in contribution.hard_mask.items() if flag and self._layer_accepts_key(layer, str(target))
                    ],
                    posterior=dict(contribution.posterior),
                    peak_clusters=list(contribution.peak_clusters),
                    compromise_template_prior=dict(contribution.compromise_template_prior),
                    confidence_raw=contribution.confidence,
                    confidence_calibrated=contribution.confidence_calibrated if contribution.confidence_calibrated is not None else contribution.confidence,
                    normalization_reason=f"{(contribution.projection or EnergyProjectionSpec(module_type=contribution.module_type, target_space=layer)).normalization_strategy}_temperature={round(float((contribution.projection or EnergyProjectionSpec(module_type=contribution.module_type, target_space=layer)).module_temperature), 4)}",
                    projection_reason=contribution.projection_reason or contribution.native_operator,
                    trace_reason=contribution.trace_reason,
                    dependency_trace=list(contribution.dependency_trace),
                    stats=stats,
                )
            )

        final_energy = {target: round(float(base_energy.get(target, 0.0)) + aggregated_normalized.get(target, 0.0), 6) for target in set(base_energy) | set(aggregated_normalized)}
        for target in hard_masked:
            final_energy[target] = float("-inf")
        winner_target = ""
        finite_targets = {target: value for target, value in final_energy.items() if value != float("-inf")}
        if finite_targets:
            winner_target = max(finite_targets, key=finite_targets.get)
        winner_posterior = _posterior_from_energy(final_energy)
        counterfactual_top_peaks = _top_peaks(final_energy, winner_posterior, hard_masked)

        return ProbabilityLayerState(
            layer=layer,
            base_energy=dict(base_energy),
            aggregated_raw_signal=aggregated_raw_signal,
            aggregated_modulated_delta=aggregated_modulated_delta,
            aggregated_inhibitory_drive=aggregated_inhibitory_drive,
            aggregated_projected_delta=aggregated_projected,
            aggregated_normalized_delta=aggregated_normalized,
            final_energy=final_energy,
            failure_taxonomy=sorted(failure_taxonomy),
            hard_masked_targets=sorted(hard_masked),
            winner_target=winner_target,
            winner_posterior=winner_posterior,
            counterfactual_top_peaks=counterfactual_top_peaks,
            contribution_audit=audits,
        )

    def _validate_token_action_alignment(
        self,
        base_energy: dict[str, float],
        contributions: list[ProbabilisticContribution],
    ) -> None:
        allowed_action_tokens = {
            str(target)
            for target in dict(base_energy).keys()
            if str(target).startswith("act:")
        }
        if not allowed_action_tokens:
            return
        illegal: dict[str, list[str]] = {}
        for contribution in contributions:
            touched_tokens = {
                key
                for key in {
                    *self._raw_signal(contribution).keys(),
                    *self._modulated_delta(contribution).keys(),
                    *self._inhibitory_drive(contribution).keys(),
                    *contribution.hard_mask.keys(),
                }
                if str(key).startswith("act:")
            }
            illegal_tokens = sorted(token for token in touched_tokens if token not in allowed_action_tokens)
            if illegal_tokens:
                illegal[contribution.module_name] = illegal_tokens
        if illegal:
            formatted = ", ".join(
                f"{module}={','.join(tokens)}"
                for module, tokens in sorted(illegal.items())
            )
            raise ValueError(f"token contributions cannot drift outside the action field: {formatted}")

    def _raw_signal(self, contribution: ProbabilisticContribution) -> dict[str, float]:
        if contribution.raw_signal:
            return dict(contribution.raw_signal)
        return dict(contribution.modulated_delta)

    def _modulated_delta(self, contribution: ProbabilisticContribution) -> dict[str, float]:
        if contribution.modulated_delta:
            return dict(contribution.modulated_delta)
        return dict(contribution.raw_signal)

    def _inhibitory_drive(self, contribution: ProbabilisticContribution) -> dict[str, float]:
        if contribution.inhibitory_drive:
            return {str(target): abs(float(value)) for target, value in contribution.inhibitory_drive.items()}
        return {}

    def _project(self, contribution: ProbabilisticContribution) -> dict[str, float]:
        projected = _merge_delta_maps(self._modulated_delta(contribution))
        for target, value in self._inhibitory_drive(contribution).items():
            projected[target] = round(projected.get(target, 0.0) - abs(float(value)), 6)
        return projected
