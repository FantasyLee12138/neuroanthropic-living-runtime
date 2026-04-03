from __future__ import annotations

from nalr.schemas.models import RenderPlan


def _clean_summary(summary: str) -> str:
    return summary.strip().strip("。！？!?；;，, ")


def _remember_clause(summary: str) -> str:
    if any(token in summary for token in ("晚饭", "吃面", "面")):
        return "把你提到的晚饭线索先记住"
    return "把这条信息先记住"


def _build_followup(plan: RenderPlan) -> str:
    summary = _clean_summary(plan.event_summary)
    if not summary:
        return ""

    wants_plan = any(token in summary for token in ("规划", "计划", "安排", "plan"))
    wants_memory = any(token in summary for token in ("记住", "记下", "提醒", "remember", "recall"))
    wants_reply = any(token in summary for token in ("回复", "回", "reply"))

    if wants_plan and wants_memory:
        return f"我先把今晚的安排理一下，再{_remember_clause(summary)}。"
    if wants_plan:
        return "我先把接下来的安排理一下。"
    if wants_memory:
        return f"我先{_remember_clause(summary)}。"
    if wants_reply:
        return "我先把回复的方向和措辞稳一下。"
    return "我先顺着你刚才提到的内容继续往下接。"


def _opening_from_expression(plan: RenderPlan) -> str:
    expression = plan.expression
    if expression.directness_level >= 0.72:
        return "我先说重点。"
    if expression.hedging_level >= 0.55 or plan.safety_constraints.get("conflict_hot"):
        return "如果你愿意，我先把这件事说得更稳一点。"
    if expression.warmth_level >= 0.62:
        return "我先接住你的意思。"
    return ""


def _action_line(plan: RenderPlan) -> str:
    expression = plan.expression
    if plan.action == "plan":
        return "我先给你一个可执行的短计划。"
    if plan.action == "recall":
        return "我先把相关线索和记忆证据提出来。"
    if plan.action == "clarify":
        return "我先确认一下你真正想解决的点。"
    if plan.action == "connect":
        return "我先接住你的感受，再继续往下说。"
    if plan.action == "rest":
        return "我先把强度收一点，再继续回应。"
    if plan.action == "wander":
        return "我先停一下，把散开的点收回来。"
    if expression.directness_level >= 0.72:
        return "先给你一个能直接发出去的回应方向。"
    return _build_followup(plan) or "我先顺着你刚才提到的内容继续回应。"


def _safety_line(plan: RenderPlan) -> str:
    relation_risk = float(plan.relation_state.get("relationship_risk", 0.0))
    reaction_risk = float(plan.perspective.get("reaction_hypothesis", {}).get("risk", 0.0))
    expression = plan.expression
    if relation_risk >= 0.65 or reaction_risk >= 0.65 or plan.safety_constraints.get("conflict_hot"):
        repair = "如果哪里不对，你可以马上打断我，我立刻调整。" if expression.repair_tendency >= 0.55 else "我会尽量把话说稳一点。"
        return repair
    if expression.warmth_level >= 0.62:
        return "我会尽量把话说得更贴近你现在的状态。"
    return ""


def fallback_render_text(plan: RenderPlan) -> str:
    summary = _clean_summary(plan.event_summary)
    context_text = f"你刚才提到“{summary}”。 " if summary else ""
    parts = [part for part in (_opening_from_expression(plan), _action_line(plan), _safety_line(plan)) if part]
    text = f"{context_text}{' '.join(parts)}".strip()
    return text or "我先顺着你刚才提到的内容继续回应。"
