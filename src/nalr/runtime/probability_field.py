from __future__ import annotations

from collections import defaultdict
from typing import Any

from nalr.schemas.models import (
    PeakArbitrationRecord,
    ProbabilisticContribution,
    ProbabilityFieldSnapshot,
    ProbabilityLayerState,
    TokenContributionTrace,
)


def _normalize_scores(values: dict[str, float]) -> dict[str, float]:
    positive = {key: max(float(value), 0.0) for key, value in values.items()}
    total = sum(positive.values()) or 1.0
    return {key: round(value / total, 6) for key, value in sorted(positive.items())}


def _merge_maps(rows: list[ProbabilisticContribution], attr: str) -> dict[str, float]:
    merged: dict[str, float] = defaultdict(float)
    for row in rows:
        for key, value in getattr(row, attr).items():
            merged[str(key)] += float(value)
    return {key: round(value, 6) for key, value in sorted(merged.items())}


def _layer_state(name: str, rows: list[ProbabilisticContribution], combined_bias: dict[str, float]) -> ProbabilityLayerState:
    suppressed: set[str] = set()
    for row in rows:
        suppressed.update(row.hard_mask)
        suppressed.update(target for target, value in row.soft_mask.items() if value < 0.0)
    return ProbabilityLayerState(
        layer=name,
        contributions=[row.module_name for row in rows],
        combined_bias=combined_bias,
        suppressed_targets=sorted(suppressed),
    )


class ProbabilityFieldIntegrator:
    def integrate(
        self,
        *,
        base_action_logits: dict[str, float],
        base_token_logits: dict[str, float],
        context_contributions: list[ProbabilisticContribution],
        memory_contributions: list[ProbabilisticContribution],
        action_contributions: list[ProbabilisticContribution],
        token_contributions: list[ProbabilisticContribution],
    ) -> ProbabilityFieldSnapshot:
        context_bias = _merge_maps(context_contributions, "attention_bias")
        memory_prior = _merge_maps(memory_contributions, "posterior")
        action_bias = _merge_maps(action_contributions + memory_contributions, "delta_logits")
        token_bias = _merge_maps(token_contributions, "delta_logits")
        action_mask = _merge_maps(action_contributions, "soft_mask")
        token_mask = _merge_maps(token_contributions, "soft_mask")

        action_logits_final = {
            action: round(float(base_action_logits.get(action, 0.0)) + action_bias.get(action, 0.0) + action_mask.get(action, 0.0), 6)
            for action in sorted({*base_action_logits, *action_bias, *action_mask})
        }
        token_logits_final = {
            token: round(float(base_token_logits.get(token, 0.0)) + token_bias.get(token, 0.0) + token_mask.get(token, 0.0), 6)
            for token in sorted({*base_token_logits, *token_bias, *token_mask})
        }

        for row in action_contributions:
            for target in row.hard_mask:
                action_logits_final[target] = min(action_logits_final.get(target, 0.0), 0.0)
        for row in token_contributions:
            for target in row.hard_mask:
                token_logits_final[target] = min(token_logits_final.get(target, 0.0), 0.0)

        action_posterior = _normalize_scores(action_logits_final)
        top_action = max(action_posterior, key=action_posterior.get) if action_posterior else "respond"
        competing = {key: value for key, value in action_posterior.items() if key != top_action}
        competing = dict(sorted(competing.items(), key=lambda item: item[1], reverse=True)[:3])
        token_traces = [
            TokenContributionTrace(
                token=token,
                final_logit=final_logit,
                module_deltas={
                    row.module_name: round(row.delta_logits.get(token, 0.0), 6)
                    for row in token_contributions
                    if token in row.delta_logits
                },
                suppressed_by=sorted(
                    {
                        row.module_name
                        for row in token_contributions
                        if token in row.hard_mask or row.soft_mask.get(token, 0.0) < 0.0
                    }
                ),
                reason=next((row.trace_reason for row in token_contributions if token in row.delta_logits), ""),
            )
            for token, final_logit in token_logits_final.items()
            if any(token in row.delta_logits or token in row.hard_mask or token in row.soft_mask for row in token_contributions)
        ]
        counterfactual = [
            {"action": action, "score": score}
            for action, score in sorted(competing.items(), key=lambda item: item[1], reverse=True)
        ]
        peak_arbitration = PeakArbitrationRecord(
            winning_peak=top_action,
            winning_score=action_posterior.get(top_action, 0.0),
            competing_peaks=competing,
            suppression_reasons={
                action: "suppressed by negative bias or guard"
                for action, score in competing.items()
                if score < action_posterior.get(top_action, 0.0)
            },
            compromise_applied=False,
        )
        return ProbabilityFieldSnapshot(
            layers=[
                _layer_state("context", context_contributions, context_bias),
                _layer_state("memory", memory_contributions, memory_prior),
                _layer_state("action", action_contributions, action_bias),
                _layer_state("token", token_contributions, token_bias),
            ],
            context_attn_final=context_bias,
            memory_prior_final=memory_prior,
            action_logits_final=action_logits_final,
            token_logits_final=token_logits_final,
            winner_posterior={
                "top_action": top_action,
                "probability": action_posterior.get(top_action, 0.0),
                "distribution": action_posterior,
            },
            counterfactual_top_peaks=counterfactual,
            token_traces=token_traces,
            peak_arbitration=peak_arbitration,
        )

