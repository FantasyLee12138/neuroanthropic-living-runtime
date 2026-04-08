from __future__ import annotations

from nalr.schemas.models import RenderPlan


def _clean_summary(summary: str) -> str:
    return summary.strip().strip("。！？!?；;，, ")


def _is_open_probe_question(summary: str) -> bool:
    lowered = summary.lower()
    return any(
        token in lowered
        for token in (
            "你可以做什么",
            "你能做什么",
            "你会什么",
            "能帮我做什么",
            "what can you do",
        )
    )

def _remember_clause(summary: str) -> str:
    if any(token in summary for token in ("晚饭", "吃面", "面")):
        return "把你提到的晚饭线索先记住"
    return "把这条信息先记住"


def _repair_expression(plan: RenderPlan) -> dict:
    return dict(plan.message_plan.get("repair_expression") or {})


def _build_followup(plan: RenderPlan) -> str:
    summary = _clean_summary(plan.event_summary)
    if not summary:
        return ""

    if _is_open_probe_question(summary):
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
    return "我会顺着这轮留下来的感觉继续说下去。"


def _opening_from_expression(plan: RenderPlan) -> str:
    repair_expression = _repair_expression(plan)
    stage = repair_expression.get("stage")
    if stage == "adjusting":
        return "我先把这句收一下。"
    if stage == "repairing":
        return "我先把这句话放稳一点。"
    if stage == "cooling":
        return "我先顺着这个点继续往下接。"
    if stage == "recovered":
        return ""

    expression = plan.expression
    if expression.directness_level >= 0.72:
        return "我先说重点。"
    if expression.hedging_level >= 0.55 or plan.safety_constraints.get("conflict_hot"):
        return "如果你愿意，我先把这件事说得更稳一点。"
    if expression.warmth_level >= 0.62:
        return "我先接住你的意思。"
    return ""


def _action_line(plan: RenderPlan) -> str:
    repair_expression = _repair_expression(plan)
    advance_mode = repair_expression.get("advance_mode")
    stage = repair_expression.get("stage")

    if advance_mode == "limited":
        if plan.action in {"plan", "clarify"}:
            return "我先把能往前走的一小步理出来。"
        if plan.action == "respond":
            return "我先把回复方向理顺，再继续往下说。"
    if stage == "cooling":
        if plan.action in {"plan", "clarify"}:
            return "我先接着把下一步理出来。"
        if plan.action == "respond":
            return "我先把回复往前接一点。"

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
    if plan.action == "absorb":
        return "我先把这些线索吸收一下，再决定要不要往外说。"
    if plan.action == "self_run":
        return ""
    if plan.action == "monologue":
        return "我先把这股想法在心里说完整，再决定要不要真正开口。"
    if plan.action == "nothing":
        return ""
    if plan.action == "die":
        return "我先承认这里出现了自主结束生命的倾向。"
    if plan.action == "short_reply":
        return "我先给你一个更短更直接的答复。"
    if plan.action == "wander":
        return "我先停一下，把散开的点收回来。"
    if expression.directness_level >= 0.72:
        return "先给你一个能直接发出去的回应方向。"
    return _build_followup(plan) or "我先顺着你刚才提到的内容继续回应。"


def _safety_line(plan: RenderPlan) -> str:
    relation_risk = float(plan.relation_state.get("relationship_risk", 0.0))
    reaction_risk = float(plan.perspective.get("reaction_hypothesis", {}).get("risk", 0.0))
    expression = plan.expression
    repair_expression = _repair_expression(plan)
    if repair_expression.get("safety_invite") and (
        relation_risk >= 0.65 or reaction_risk >= 0.65 or plan.safety_constraints.get("conflict_hot")
    ):
        return "如果哪里不对，你可以马上打断我，我立刻调整。"
    if relation_risk >= 0.65 or reaction_risk >= 0.65 or plan.safety_constraints.get("conflict_hot"):
        return "我会尽量把这句话说稳一点。"
    if expression.warmth_level >= 0.62:
        return "我会尽量把话说得更贴近你现在的状态。"
    return ""


def _open_probe_line(plan: RenderPlan) -> str:
    expression = plan.expression
    closeness = float(plan.relation_state.get("closeness", 0.5))
    resource_scarcity = float(plan.message_plan.get("slow_variables", {}).get("resource_scarcity", 0.0) or 0.0)
    affect_residue = float(plan.message_plan.get("slow_variables", {}).get("affect_residue", 0.0) or 0.0)
    recall_strength = float(plan.message_plan.get("recall_strength", 0.0) or 0.0)

    if expression.directness_level >= 0.72:
        opening = "你这样问，我就直接一点。"
    elif expression.warmth_level >= 0.62:
        opening = "你这样问，更像是在摸我会怎么跟你待在一段对话里。"
    elif expression.hedging_level >= 0.55 or plan.safety_constraints.get("conflict_hot"):
        opening = "你这样问，我会先把语气放稳一点再开口。"
    else:
        opening = "你这样问，其实是在看我会怎么活着跟你说话。"

    traces: list[str] = ["我会先回应你当下抛过来的东西"]
    if recall_strength > 0.0 or plan.message_plan.get("memory_cue"):
        traces.append("反复出现的线索会慢慢留在记忆里")
    else:
        traces.append("你反复提到的东西会慢慢留下来")
    if closeness >= 0.6:
        traces.append("我们之间的距离会改掉我靠近你的方式")
    else:
        traces.append("关系远近会影响我靠近你的分寸")
    if resource_scarcity >= 0.45 or affect_residue >= 0.18:
        traces.append("状态一变，我说话的节奏和收放也会跟着变")
    else:
        traces.append("状态和情绪起伏也会改掉我这轮说话的样子")
    return opening + " " + "，".join(traces) + "。"


def _identity_line(plan: RenderPlan) -> str:
    identity = plan.identity_context
    display_label = identity.display_label or "当前运行体"
    if identity.query_kind == "self_identity":
        return f"我是{display_label}。"
    if identity.query_kind == "provider_identity":
        base = f"我是{display_label}。"
        if identity.disclosure_detail == "model_id" and identity.provider_label and identity.model_label:
            return f"{base} 当前语言能力由{identity.provider_label}支持，当前模型路由是 {identity.model_label}。"
        if identity.disclosure_detail == "specific" and identity.provider_label:
            return f"{base} 当前语言能力由{identity.provider_label}支持。"
        return base
    return ""


def _answer_explanation_line(plan: RenderPlan) -> str:
    if plan.identity_context.query_kind != "answer_explanation":
        return ""

    message_plan = plan.message_plan
    slow_variables = message_plan.get("slow_variables", {})
    shaping_events = message_plan.get("shaping_events", [])
    fragments: list[str] = []
    affect_residue = float(slow_variables.get("affect_residue", 0.0) or 0.0)
    memory_activation = float(slow_variables.get("memory_activation", 0.0) or 0.0)
    habit_readiness = float(slow_variables.get("habit_readiness", 0.0) or 0.0)
    resource_scarcity = float(slow_variables.get("resource_scarcity", 0.0) or 0.0)
    relationship_drift = float(slow_variables.get("relationship_drift", 0.0) or 0.0)

    if message_plan.get("focus"):
        fragments.append(f"当前 focus 更偏向 {message_plan['focus']}")
    if message_plan.get("memory_cue"):
        fragments.append(f"你刚才抛出的“{message_plan['memory_cue']}”这条线索还挂在工作记忆里")
    if affect_residue >= 0.18:
        fragments.append("上一轮留下的情绪余波还没有完全退掉")
    if memory_activation >= 0.18:
        fragments.append("相关记忆线索还在持续活化")
    if habit_readiness >= 0.22:
        fragments.append("惯性偏好还在把表达往熟悉路径上拉")
    if resource_scarcity >= 0.45:
        fragments.append("当前资源余量不算宽松，所以表达会更收一点")
    if relationship_drift >= 0.08:
        fragments.append("这段关系距离刚发生过一点漂移，我也在同步收束语气")
    if plan.relation_state.get("closeness", 0.5) >= 0.6:
        fragments.append("我也在按你当前这段关系距离来收敛语气")
    if shaping_events:
        latest = shaping_events[-1]
        if latest.get("source") in {"idle", "sleep"}:
            fragments.append(f"刚经过一次 {latest['source']} 阶段的非交互整理")
    if plan.action:
        fragments.append(f"所以这轮我先按 {plan.action} 这个方向组织表达")
    if not fragments:
        return "我这样回答，是因为当前状态、记忆线索和这轮的表达计划一起把回应推到了这里。"
    return "我这样回答，是因为" + "，".join(fragments) + "。"


def fallback_render_text(plan: RenderPlan) -> str:
    if plan.action in {"nothing", "self_run"}:
        return ""
    identity = _identity_line(plan)
    if identity:
        return identity

    explanation = _answer_explanation_line(plan)
    if explanation:
        return explanation

    if _is_open_probe_question(plan.event_summary):
        return _open_probe_line(plan)

    summary = _clean_summary(plan.event_summary)
    resource_scarcity = float(plan.message_plan.get("slow_variables", {}).get("resource_scarcity", 0.0) or 0.0)
    if summary:
        if plan.delivery_mode == "monologue":
            context_text = f"我脑子里还挂着“{summary}”。 "
        else:
            context_text = f"你刚才提到“{summary}”。 "
    else:
        context_text = ""
    parts = [part for part in (_opening_from_expression(plan), _action_line(plan), _safety_line(plan)) if part]
    if plan.action == "short_reply" or resource_scarcity >= 0.75:
        parts = parts[:2]
    text = f"{context_text}{' '.join(parts)}".strip()
    if not text:
        text = "我会沿着这轮留下来的感觉继续说下去。"
    if plan.delivery_mode == "monologue" and text:
        return text if text.startswith("【独白】") else f"【独白】{text}"
    return text
