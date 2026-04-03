from __future__ import annotations


def compute_style_profile(state: dict, scenario: str, output_styles: dict) -> dict:
    style_name = "task_focused"
    if state["body_energy"] < 0.45:
        style_name = "tired"
    elif state["focus"] == "wander":
        style_name = "distracted"
    elif scenario == "companion" and state["mood"] < 0.45:
        style_name = "guarded"

    profile = dict(output_styles.get(style_name, {}))
    profile["style_name"] = style_name
    return profile

