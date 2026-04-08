from __future__ import annotations

from typing import Any, Callable

from nalr.schemas.models import EnergyProjectionSpec, ProbabilisticContribution


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class LongRunAnalyzer:
    def __init__(self, trace_store, identity_payload: Callable[[], dict[str, Any]], core_actions: tuple[str, ...]) -> None:
        self.trace_store = trace_store
        self.identity_payload = identity_payload
        self.core_actions = core_actions

    def _conflict_arbitration_summary(self, trace: dict[str, Any]) -> dict[str, Any]:
        summary = dict(trace.get("conflict_arbitration", {}) or {})
        probability_field = dict(trace.get("probability_field", {}) or {})
        action_layer = probability_field.get("action", {}) if isinstance(probability_field, dict) else {}
        audit_rows = action_layer.get("contribution_audit", []) if isinstance(action_layer, dict) else []
        conflict_row = next(
            (
                row
                for row in audit_rows
                if isinstance(row, dict) and row.get("module_name") == "ConflictMonitorAgent"
            ),
            {},
        )
        if not conflict_row:
            return summary
        fallback = {
            "critical_conflict": bool(conflict_row.get("critical_conflict", False)),
            "circuit_breaker": dict(conflict_row.get("circuit_breaker", {}) or {}),
            "winning_priority": str(conflict_row.get("winning_priority") or ""),
            "winner_peak_posterior": dict(conflict_row.get("posterior", {}) or {}),
            "hard_masked_targets": list(conflict_row.get("hard_masked_targets", []) or []),
        }
        for key, value in fallback.items():
            if key not in summary or summary.get(key) in ({}, [], "", None):
                summary[key] = value
        return summary

    def _personality_anchor_summary(
        self,
        *,
        personality_anchor: dict[str, Any],
        top_drivers: list[dict[str, Any]],
        rounds_window: int = 12,
    ) -> dict[str, Any]:
        rounds = self.trace_store.list_rounds()[-rounds_window:]
        driver_counts: dict[str, float] = {}
        for trace in rounds:
            for driver in list(trace.get("top_drivers", []) or [])[:3]:
                if not isinstance(driver, dict):
                    continue
                action_name = str(driver.get("action_name") or "").strip()
                score = abs(float(driver.get("score", 0.0) or 0.0))
                if not action_name or score <= 0.0:
                    continue
                driver_counts[action_name] = round(driver_counts.get(action_name, 0.0) + score, 6)
        for driver in top_drivers[:3]:
            if not isinstance(driver, dict):
                continue
            action_name = str(driver.get("action_name") or "").strip()
            score = abs(float(driver.get("score", 0.0) or 0.0))
            if not action_name or score <= 0.0:
                continue
            driver_counts[action_name] = round(driver_counts.get(action_name, 0.0) + score, 6)
        dominant_actions = [
            action
            for action, _ in sorted(driver_counts.items(), key=lambda item: (item[1], item[0]), reverse=True)[:4]
        ]
        return {
            "axis_baseline": dict(personality_anchor.get("axis_baseline", {}) or {}),
            "stability": round(float(personality_anchor.get("stability", 0.0) or 0.0), 4),
            "anchor_alignment": round(float(personality_anchor.get("alignment", 0.0) or 0.0), 4),
            "anchor_drift": round(float(personality_anchor.get("drift", 0.0) or 0.0), 4),
            "evidence_anchors": list(personality_anchor.get("evidence_anchors", []) or []),
            "dominant_actions": dominant_actions,
            "driver_signature": "|".join(dominant_actions[:3]),
        }

    def build_round_projection(
        self,
        *,
        round_id: int,
        vitality_snapshot: dict[str, Any],
        authenticity: dict[str, Any],
        identity_evolution: dict[str, Any],
        shaping_events: list[dict[str, Any]],
        personality_anchor: dict[str, Any] | None = None,
        top_drivers: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        penalty = float(authenticity.get("provider_leak_penalty", 0.0)) + float(authenticity.get("false_self_claim_penalty", 0.0))
        continuity_score = _clip(float(authenticity.get("self_grounding_score", 0.0)) - penalty * 0.25, 0.0, 1.0)
        volatility_signal = max(
            float(vitality_snapshot.get("affect_residue", 0.0)),
            float(vitality_snapshot.get("relationship_drift", 0.0)),
            float(vitality_snapshot.get("resource_scarcity", 0.0)),
        )
        projection = {
            "continuity_window": min(max(round_id, 1), 8),
            "self_consistency_score": round(continuity_score, 4),
            "volatility_signal": round(volatility_signal, 4),
            "non_interactive_shift": len([item for item in shaping_events if item.get("non_interactive")]),
            "rename_reason": identity_evolution.get("rename_reason", ""),
        }
        if personality_anchor:
            projection["personality_anchor_summary"] = self._personality_anchor_summary(
                personality_anchor=personality_anchor,
                top_drivers=list(top_drivers or []),
            )
        return projection

    def build_online_projection(
        self,
        *,
        round_id: int,
        slow_variables: dict[str, Any],
        shaping_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        latest_seed_getter = getattr(self.trace_store, "latest_round_projection_seed", None)
        latest_seed = latest_seed_getter() if callable(latest_seed_getter) else {}
        if not latest_seed:
            rounds = self.trace_store.list_rounds()
            if rounds:
                latest = rounds[-1]
                latest_seed = {
                    "long_run_projection": dict(latest.get("long_run_projection", {}) or {}),
                    "authenticity": dict(latest.get("authenticity", {}) or {}),
                }
        if latest_seed:
            latest_projection = dict(latest_seed.get("long_run_projection", {}) or {})
            if "self_consistency_score" in latest_projection:
                continuity_score = _clip(float(latest_projection.get("self_consistency_score", 0.5) or 0.0), 0.0, 1.0)
            else:
                latest_authenticity = dict(latest_seed.get("authenticity", {}) or {})
                continuity_score = _clip(float(latest_authenticity.get("self_grounding_score", 0.5) or 0.0), 0.0, 1.0)
        else:
            history = self.metrics_summary()
            continuity_score = _clip(float(history.get("self_consistency_score", 0.5) or 0.0), 0.0, 1.0)
        volatility_signal = max(
            float(slow_variables.get("affect_residue", 0.0) or 0.0),
            float(slow_variables.get("relationship_drift", 0.0) or 0.0),
            float(slow_variables.get("resource_scarcity", 0.0) or 0.0),
            float(slow_variables.get("memory_activation", 0.0) or 0.0) * 0.6,
        )
        projection = {
            "continuity_window": min(max(round_id, 1), 8),
            "self_consistency_score": round(continuity_score, 4),
            "volatility_signal": round(_clip(volatility_signal, 0.0, 1.0), 4),
            "non_interactive_shift": len([item for item in shaping_events if item.get("non_interactive")]),
            "rename_reason": "",
        }
        if latest_seed:
            latest_projection = dict(latest_seed.get("long_run_projection", {}) or {})
            if latest_projection.get("personality_anchor_summary"):
                projection["personality_anchor_summary"] = dict(latest_projection.get("personality_anchor_summary", {}) or {})
        return projection

    def build_long_run_prior_contribution(self, projection: dict[str, Any]) -> ProbabilisticContribution:
        self_consistency_score = float(projection.get("self_consistency_score", 0.0) or 0.0)
        volatility_signal = float(projection.get("volatility_signal", 0.0) or 0.0)
        continuity_window = float(projection.get("continuity_window", 0.0) or 0.0)
        non_interactive_shift = float(projection.get("non_interactive_shift", 0.0) or 0.0)
        continuity_strength = _clip(self_consistency_score * 0.7 + min(continuity_window, 8.0) / 8.0 * 0.3, 0.0, 1.0)
        drift_pressure = _clip(volatility_signal * 0.7 + min(non_interactive_shift, 3.0) / 3.0 * 0.3, 0.0, 1.0)

        modulated_delta = {
            "plan": round(0.12 * continuity_strength, 6),
            "respond": round(0.08 * continuity_strength, 6),
            "clarify": round(0.05 * continuity_strength, 6),
            "wander": round(-(0.10 * continuity_strength + 0.08 * drift_pressure), 6),
        }
        if volatility_signal >= 0.12:
            modulated_delta["rest"] = round(0.04 * drift_pressure, 6)
        if non_interactive_shift > 0.0:
            modulated_delta["recall"] = round(0.03 * min(non_interactive_shift, 3.0), 6)

        confidence = _clip(0.28 + continuity_strength * 0.42 + drift_pressure * 0.18, 0.0, 1.0)
        dependency_trace = [
            f"self_consistency_score:{round(self_consistency_score, 4)}",
            f"volatility_signal:{round(volatility_signal, 4)}",
            f"continuity_window:{int(round(continuity_window))}",
            f"non_interactive_shift:{int(round(non_interactive_shift))}",
        ]
        rename_reason = str(projection.get("rename_reason", "") or "")
        if rename_reason:
            dependency_trace.append(f"rename_reason:{rename_reason}")
        return ProbabilisticContribution(
            module_name="LongRunAnalyzer",
            module_type="longrun",
            level="action",
            target_space="action",
            raw_signal=dict(modulated_delta),
            modulated_delta=modulated_delta,
            confidence=confidence,
            confidence_calibrated=round(_clip(confidence * (0.92 - volatility_signal * 0.08), 0.0, 1.0), 4),
            trace_reason=(
                f"long-run continuity prior consistency={self_consistency_score:.2f} "
                f"volatility={volatility_signal:.2f}"
            ),
            projection_reason="long-run prior projected from longitudinal continuity summary",
            applied_at_stage="long_run_prior",
            native_operator="longitudinal_prior",
            dependency_trace=dependency_trace,
            projection=EnergyProjectionSpec(module_type="longrun", target_space="action", module_temperature=0.9),
        )

    def _average(self, values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    def _estimate_affect_half_life(self, rounds: list[dict[str, Any]]) -> float:
        residues = [float(trace.get("vitality_snapshot", {}).get("affect_residue", 0.0)) for trace in rounds]
        spans: list[float] = []
        for idx, peak in enumerate(residues):
            if peak < 0.25:
                continue
            if idx > 0 and peak < residues[idx - 1]:
                continue
            target = peak / 2
            for future_idx in range(idx + 1, len(residues)):
                if residues[future_idx] <= target:
                    spans.append(float(future_idx - idx))
                    break
        return self._average(spans)

    def _estimate_recovery_duration(self, rounds: list[dict[str, Any]]) -> float:
        residues = [float(trace.get("vitality_snapshot", {}).get("affect_residue", 0.0)) for trace in rounds]
        spans: list[float] = []
        for idx, residue in enumerate(residues):
            if residue < 0.35:
                continue
            for future_idx in range(idx + 1, len(residues)):
                if residues[future_idx] <= 0.18:
                    spans.append(float(future_idx - idx))
                    break
        return self._average(spans)

    def _event_variance_metric(self, rounds: list[dict[str, Any]], bucket_fn) -> float:
        grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for trace in rounds:
            vitality = trace.get("vitality_snapshot", {})
            cue = str(vitality.get("cue") or "").strip()
            if not cue:
                continue
            bucket = str(bucket_fn(trace, vitality))
            grouped.setdefault(cue, {}).setdefault(bucket, []).append(trace)

        scores: list[float] = []
        for cue_groups in grouped.values():
            if len(cue_groups) < 2:
                continue
            actions = set()
            warmth_values: list[float] = []
            directness_values: list[float] = []
            for traces in cue_groups.values():
                for trace in traces:
                    actions.add(str(trace.get("sampled_action", "")))
                    expression = trace.get("render_plan", {}).get("expression", {})
                    warmth_values.append(float(expression.get("warmth_level", 0.0)))
                    directness_values.append(float(expression.get("directness_level", 0.0)))
            action_var = (len(actions) - 1) / max(len(self.core_actions) - 1, 1)
            warmth_var = (max(warmth_values) - min(warmth_values)) if warmth_values else 0.0
            directness_var = (max(directness_values) - min(directness_values)) if directness_values else 0.0
            scores.append(round(_clip(action_var * 0.5 + warmth_var * 0.3 + directness_var * 0.2), 4))
        return self._average(scores)

    def metrics_summary(self) -> dict[str, Any]:
        rounds = self.trace_store.list_rounds()
        sampled_actions: dict[str, int] = {}
        modes: dict[str, int] = {}
        driver_counts: dict[str, int] = {}
        budget_values: list[float] = []
        safe_mode_rounds = 0
        texts: list[str] = []
        intro_texts: list[str] = []
        provider_leak_hits = 0
        false_self_hits = 0
        rename_count = 0
        grounding_scores: list[float] = []
        cue_rounds = 0
        cue_recall_hits = 0
        gist_hits = 0
        detail_distortion_hits = 0
        interference_values: list[float] = []
        overreaction_hits = 0
        flatness_hits = 0
        habit_takeover_hits = 0
        mode_lock_values: list[float] = []
        forced_recovery_checks = 0
        forced_recovery_hits = 0
        conflict_repair_checks = 0
        conflict_repair_hits = 0
        for trace in rounds:
            sampled_actions[trace["sampled_action"]] = sampled_actions.get(trace["sampled_action"], 0) + 1
            modes[trace["mode"]] = modes.get(trace["mode"], 0) + 1
            budget_values.append(trace["state_snapshot"]["budget_remaining"])
            if trace["state_snapshot"]["safe_mode"]:
                safe_mode_rounds += 1
            for driver in trace["top_drivers"]:
                driver_counts[driver["agent_name"]] = driver_counts.get(driver["agent_name"], 0) + 1
            rendered = trace.get("rendered_expression", {})
            authenticity = trace.get("authenticity", rendered.get("authenticity", {}))
            render_plan = trace.get("render_plan", {})
            texts.append(rendered.get("text", ""))
            if render_plan.get("identity_context", {}).get("query_kind") == "self_identity":
                intro_texts.append(rendered.get("text", ""))
            provider_leak_hits += int("provider_leak" in authenticity.get("violation_types", []))
            false_self_hits += int("false_self_claim" in authenticity.get("violation_types", []))
            rename_count += int(bool(trace.get("identity_evolution", {}).get("rename_event")))
            grounding_scores.append(float(authenticity.get("self_grounding_score", 1.0)))

            vitality = trace.get("vitality_snapshot", {})
            if vitality.get("cue_present"):
                cue_rounds += 1
                memory_activation = float(vitality.get("memory_activation", 0.0))
                cue_recall_hits += int(memory_activation > 0.0)
                gist_hits += int(memory_activation >= 0.18)
                detail_distortion_hits += int(float(vitality.get("memory_interference", 0.0)) >= 0.12 and not bool(vitality.get("detail_available")))
                interference_values.append(float(vitality.get("memory_interference", 0.0)))
            overreaction_hits += int(float(vitality.get("affect_residue", 0.0)) > abs(float(vitality.get("trigger_valence", 0.0))) + 0.25)
            habit_takeover_hits += int(bool(vitality.get("habit_takeover")))
            mode_lock_values.append(float(trace.get("state_snapshot", {}).get("focus_lock_count", 0.0)))

        for idx in range(1, len(rounds)):
            prev_trace = rounds[idx - 1]
            trace = rounds[idx]
            prev_vitality = prev_trace.get("vitality_snapshot", {})
            trace_vitality = trace.get("vitality_snapshot", {})
            mood_delta = abs(float(trace.get("state_snapshot", {}).get("mood", 0.0)) - float(prev_trace.get("state_snapshot", {}).get("mood", 0.0)))
            if abs(float(trace_vitality.get("trigger_valence", 0.0))) >= 0.25 and float(trace_vitality.get("affect_residue", 0.0)) < 0.08 and mood_delta < 0.015:
                flatness_hits += 1
            if any(item.get("stage") == "forced_mode_switch" for item in prev_trace.get("gate_decisions", [])):
                forced_recovery_checks += 1
                forced_recovery_hits += int(trace.get("sampled_action") != prev_trace.get("sampled_action"))
            prev_conflict = self._conflict_arbitration_summary(prev_trace)
            prev_critical = bool(prev_conflict.get("critical_conflict")) or bool(prev_conflict.get("circuit_breaker", {}).get("active"))
            if prev_critical:
                conflict_repair_checks += 1
                repair_tendency = float(trace.get("render_plan", {}).get("expression", {}).get("repair_tendency", 0.0))
                conflict_repair_hits += int(repair_tendency >= 0.40)
        top_agents = [{"agent_name": name, "count": count} for name, count in sorted(driver_counts.items(), key=lambda item: item[1], reverse=True)[:5]]
        avg_budget = round(sum(budget_values) / len(budget_values), 4) if budget_values else 0.0
        transitions = sum(1 for idx in range(1, len(texts)) if texts[idx] == texts[idx - 1])
        surface_repeat_rate = round(transitions / max(len(texts) - 1, 1), 4) if texts else 0.0
        intro_repeats = sum(1 for idx in range(1, len(intro_texts)) if intro_texts[idx] == intro_texts[idx - 1])
        intro_template_reuse_rate = round(intro_repeats / max(len(intro_texts) - 1, 1), 4) if intro_texts else 0.0
        identity = self.identity_payload()
        total_rounds = len(rounds)
        avg_grounding = round(sum(grounding_scores) / len(grounding_scores), 4) if grounding_scores else 1.0
        issue_penalty = (provider_leak_hits + false_self_hits + rename_count) / max(total_rounds, 1) if total_rounds else 0.0
        self_consistency_score = round(_clip(avg_grounding - issue_penalty * 0.25), 4)
        low_diversity_alert = surface_repeat_rate >= 0.55 or intro_template_reuse_rate >= 0.45
        return {
            "total_rounds": total_rounds,
            "sampled_actions": sampled_actions,
            "mode_counts": modes,
            "safe_mode_rounds": safe_mode_rounds,
            "average_budget_remaining": avg_budget,
            "top_agents": top_agents,
            "current_display_name": identity.get("display_name"),
            "alias_count": len(identity.get("aliases", [])),
            "rename_count": rename_count,
            "self_consistency_score": self_consistency_score,
            "provider_leak_rate": round(provider_leak_hits / max(total_rounds, 1), 4) if total_rounds else 0.0,
            "false_self_claim_rate": round(false_self_hits / max(total_rounds, 1), 4) if total_rounds else 0.0,
            "surface_repeat_rate": surface_repeat_rate,
            "intro_template_reuse_rate": intro_template_reuse_rate,
            "low_diversity_alert": low_diversity_alert,
            "cue_recall_success_rate": round(cue_recall_hits / max(cue_rounds, 1), 4) if cue_rounds else 0.0,
            "gist_preservation_rate": round(gist_hits / max(cue_rounds, 1), 4) if cue_rounds else 0.0,
            "detail_distortion_rate": round(detail_distortion_hits / max(cue_rounds, 1), 4) if cue_rounds else 0.0,
            "memory_interference_rate": self._average(interference_values),
            "affect_residue_half_life": self._estimate_affect_half_life(rounds),
            "recovery_duration": self._estimate_recovery_duration(rounds),
            "overreaction_frequency": round(overreaction_hits / max(total_rounds, 1), 4) if total_rounds else 0.0,
            "flatness_rate": round(flatness_hits / max(len(rounds) - 1, 1), 4) if len(rounds) > 1 else 0.0,
            "habit_takeover_rate": round(habit_takeover_hits / max(total_rounds, 1), 4) if total_rounds else 0.0,
            "mode_lock_duration": self._average(mode_lock_values),
            "forced_recovery_success_rate": round(forced_recovery_hits / max(forced_recovery_checks, 1), 4) if forced_recovery_checks else 0.0,
            "post_conflict_repair_rate": round(conflict_repair_hits / max(conflict_repair_checks, 1), 4) if conflict_repair_checks else 0.0,
            "same_event_cross_context_variance": self._event_variance_metric(rounds, lambda trace, vitality: trace.get("scenario", "")),
            "same_event_cross_relation_variance": self._event_variance_metric(rounds, lambda trace, vitality: round(float(vitality.get("relationship_closeness", 0.5)) / 0.25)),
            "same_event_cross_resource_variance": self._event_variance_metric(rounds, lambda trace, vitality: round(float(vitality.get("resource_scarcity", 0.0)) / 0.25)),
        }

    def authenticity_timeline(self) -> dict[str, Any]:
        points = []
        for trace in self.trace_store.list_rounds():
            identity_context = trace.get("render_plan", {}).get("identity_context", {})
            authenticity = trace.get("authenticity", trace.get("rendered_expression", {}).get("authenticity", {}))
            identity_evolution = trace.get("identity_evolution", {})
            points.append(
                {
                    "round_id": trace["round_id"],
                    "query_kind": identity_context.get("query_kind", "general"),
                    "query_intent": identity_context.get("query_intent", "general_exchange"),
                    "display_name": identity_context.get("display_label"),
                    "provider_leak_detected": authenticity.get("provider_leak_detected", False),
                    "false_self_claim_detected": authenticity.get("false_self_claim_detected", False),
                    "self_grounding_score": authenticity.get("self_grounding_score", 0.0),
                    "provider_leak_penalty": authenticity.get("provider_leak_penalty", 0.0),
                    "false_self_claim_penalty": authenticity.get("false_self_claim_penalty", 0.0),
                    "guard_action": authenticity.get("guard_action", "pass"),
                    "disclosure_detail": authenticity.get("disclosure_detail", identity_context.get("disclosure_detail", "none")),
                    "disclosure_intent": identity_context.get("disclosure_intent", "withhold"),
                    "rename_event": identity_evolution.get("rename_event"),
                    "rename_reason": identity_evolution.get("rename_reason", ""),
                    "state_sources": authenticity.get("state_sources", []),
                    "identity_shaping_sources": identity_evolution.get("identity_shaping_sources", []),
                }
            )
        return {"points": points}

    def vitality_timeline(self) -> dict[str, Any]:
        points = []
        for trace in self.trace_store.list_rounds():
            vitality = trace.get("vitality_snapshot", {})
            non_interactive_events = trace.get("vitality_events", [])
            summary = [str(item.get("shaping_detail")) for item in non_interactive_events if item.get("non_interactive") and item.get("shaping_detail")]
            points.append(
                {
                    "round_id": trace["round_id"],
                    "scenario": trace.get("scenario"),
                    "mode": trace.get("mode"),
                    "affect_residue": vitality.get("affect_residue", 0.0),
                    "memory_activation": vitality.get("memory_activation", 0.0),
                    "memory_interference": vitality.get("memory_interference", 0.0),
                    "habit_readiness": vitality.get("habit_readiness", 0.0),
                    "resource_scarcity": vitality.get("resource_scarcity", 0.0),
                    "relationship_drift": vitality.get("relationship_drift", 0.0),
                    "non_interactive_events": non_interactive_events,
                    "non_interactive_summary": summary,
                }
            )
        return {"points": points}
