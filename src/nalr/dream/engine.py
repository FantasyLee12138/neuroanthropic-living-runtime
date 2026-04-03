from __future__ import annotations

from uuid import uuid4

from nalr.schemas.models import DreamProposalBundle, DreamRunRequest, DreamTrace, to_dict


class OneiroiAgent:
    def run(self, request: DreamRunRequest) -> dict:
        snapshot = request.snapshot
        policy_allowed = list(request.policy.get("allowed_types", []))
        hot = list(snapshot.memory_refs.get("hot", []))
        warm = list(snapshot.memory_refs.get("warm", []))
        cue = (hot[0] if hot else (warm[0] if warm else None)) or "drift"
        allow_identity = "limited_identity_drift" in policy_allowed and bool(request.budget.get("allow_identity_drift"))
        route = "heuristic_fallback"

        bundle = DreamProposalBundle(
            memory_consolidation=[
                {
                    "cue": cue,
                    "gist_delta": 0.045 if snapshot.trigger == "sleep_full" else 0.02,
                    "detail_delta": -0.02 if snapshot.trigger == "sleep_full" else -0.005,
                    "interference_scale": 0.88 if snapshot.trigger == "sleep_full" else 0.95,
                    "stable_prior_delta": 0.03 if snapshot.trigger == "sleep_full" else 0.012,
                }
            ],
            emotion_adjustments=[
                {"field": "affect_residue", "multiplier": 0.82 if snapshot.mode == "sleep" else 0.92},
                {"field": "mood", "target": 0.55, "mix": 0.10 if snapshot.mode == "sleep" else 0.04},
            ],
            habit_adjustments=[{"pattern": cue, "delta": 0.022 if snapshot.mode == "sleep" else 0.008}],
            dream_memory_write=(
                [{"cue": cue, "gist": f"dream::{snapshot.mode}::{cue}"}]
                if snapshot.trigger == "sleep_full"
                else []
            ),
            relationship_adjustments=(
                [{"target": "user", "trust_delta": 0.01}]
                if snapshot.trigger == "sleep_full" and float(snapshot.relationship_state.get("closeness", 0.5)) >= 0.55
                else []
            ),
            limited_identity_drift=(
                [
                    {
                        "credit_delta": 0.18,
                        "evidence": list(snapshot.identity_evidence_summary.get("anchors", []))[:2],
                        "proposed_alias": cue,
                    }
                ]
                if allow_identity and snapshot.trigger == "sleep_full"
                else []
            ),
        )
        trace = DreamTrace(
            dream_run_id=f"dream-{uuid4().hex[:12]}",
            sleep_session_id=snapshot.sleep_session_id,
            mode=snapshot.mode,
            trigger=snapshot.trigger,
            route=route,
            degraded=True,
            evaluated_types=policy_allowed,
            proposal_counts={
                "memory_consolidation": len(bundle.memory_consolidation),
                "emotion_adjustments": len(bundle.emotion_adjustments),
                "habit_adjustments": len(bundle.habit_adjustments),
                "dream_memory_write": len(bundle.dream_memory_write),
                "relationship_adjustments": len(bundle.relationship_adjustments),
                "limited_identity_drift": len(bundle.limited_identity_drift),
            },
            cue=cue,
            dominant_emotion=str(snapshot.emotional_baseline.get("dominant_emotion", "")),
        )
        return {"proposal_bundle": to_dict(bundle), "trace": to_dict(trace)}
