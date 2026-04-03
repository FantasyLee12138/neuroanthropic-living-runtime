from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from nalr.cil.runtime import CommandInterfaceLayer
from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent, to_dict


app = typer.Typer(help="NeuroAnthropic Living Runtime CLI")
state_app = typer.Typer()
body_app = typer.Typer()
mood_app = typer.Typer()
relation_app = typer.Typer()
focus_app = typer.Typer()
memory_app = typer.Typer()
habit_app = typer.Typer()
mode_app = typer.Typer()
trace_app = typer.Typer()
agent_app = typer.Typer()
skill_app = typer.Typer()
checkpoint_app = typer.Typer()
safe_app = typer.Typer()
budget_app = typer.Typer()
debug_app = typer.Typer()
replay_app = typer.Typer()
why_app = typer.Typer()
what_app = typer.Typer()
eval_app = typer.Typer()

app.add_typer(state_app, name="state")
app.add_typer(body_app, name="body")
app.add_typer(mood_app, name="mood")
app.add_typer(relation_app, name="relation")
app.add_typer(focus_app, name="focus")
app.add_typer(memory_app, name="memory")
app.add_typer(habit_app, name="habit")
app.add_typer(mode_app, name="mode")
app.add_typer(trace_app, name="trace")
app.add_typer(agent_app, name="agent")
app.add_typer(skill_app, name="skill")
app.add_typer(checkpoint_app, name="checkpoint")
app.add_typer(safe_app, name="safe")
app.add_typer(budget_app, name="budget")
app.add_typer(debug_app, name="debug")
app.add_typer(replay_app, name="replay")
app.add_typer(why_app, name="why")
app.add_typer(what_app, name="what")
app.add_typer(eval_app, name="eval")


def get_controller() -> RuntimeController:
    home = Path(os.environ.get("NALR_HOME", ".alive"))
    config_dir = Path(os.environ.get("NALR_CONFIG_DIR", "config"))
    project_root = home.parent if home.name == ".alive" else Path.cwd()
    return RuntimeController(project_root=project_root, config_root=config_dir, home_path=home)


def get_cil() -> CommandInterfaceLayer:
    return CommandInterfaceLayer(get_controller())


def emit(payload: object) -> None:
    typer.echo(json.dumps(to_dict(payload), ensure_ascii=False, indent=2))


@state_app.command("show")
def state_show() -> None:
    emit(get_cil().show_state())


@body_app.command("show")
def body_show() -> None:
    emit(get_cil().show_body())


@body_app.command("rest")
def body_rest() -> None:
    emit(get_cil().apply("body rest"))


@mood_app.command("show")
def mood_show() -> None:
    emit(get_cil().show_mood())


@mood_app.command("calm")
def mood_calm() -> None:
    emit(get_cil().apply("mood calm"))


@relation_app.command("show")
def relation_show(target: str) -> None:
    emit(get_cil().show_relation(target))


@focus_app.command("show")
def focus_show() -> None:
    emit(get_cil().show_focus())


@memory_app.command("top")
def memory_top(limit: int = 5) -> None:
    emit(get_cil().memory_top(limit))


@habit_app.command("top")
def habit_top(limit: int = 5) -> None:
    emit(get_cil().habit_top(limit))


@mode_app.command("set")
def mode_set(mode_name: str) -> None:
    emit(get_controller().apply_command(f"mode set {mode_name}"))


@trace_app.command("round")
def trace_round(round_id: int) -> None:
    emit(get_controller().trace_round(round_id))


@trace_app.command("why")
def trace_why(round_id: int) -> None:
    emit(get_controller().why_this(round_id))


@trace_app.command("contribution")
def trace_contribution(round_id: int) -> None:
    emit(get_controller().contribution_breakdown(round_id))


@trace_app.command("compact")
def trace_compact() -> None:
    emit(get_controller().compact_traces())


@agent_app.command("list")
def agent_list() -> None:
    emit(get_controller().agent_list())


@agent_app.command("disable")
def agent_disable(agent_name: str) -> None:
    emit(get_controller().apply_command(f"agent disable {agent_name}"))


@agent_app.command("enable")
def agent_enable(agent_name: str) -> None:
    emit(get_controller().apply_command(f"agent enable {agent_name}"))


@skill_app.command("stats")
def skill_stats() -> None:
    emit(get_cil().skill_stats())


@skill_app.command("profile")
def skill_profile(skill_name: str) -> None:
    emit(get_cil().skill_profile(skill_name))


@checkpoint_app.command("create")
def checkpoint_create() -> None:
    emit(get_controller().checkpoint())


@checkpoint_app.command("rewind")
def checkpoint_rewind(checkpoint_id: str) -> None:
    emit(get_controller().rewind(checkpoint_id))


@safe_app.command("on")
def safe_on() -> None:
    emit(get_controller().apply_command("safe on"))


@safe_app.command("off")
def safe_off() -> None:
    emit(get_controller().apply_command("safe off"))


@budget_app.command("show")
def budget_show() -> None:
    emit({"budget_remaining": get_controller().load_runtime_state().budget_remaining})


@debug_app.command("weight")
def debug_weight(agent_name: str, weight: float) -> None:
    emit(get_controller().apply_command(f"debug weight {agent_name} {weight}"))


@replay_app.command("round")
def replay_round(round_id: int, seed: int = 0) -> None:
    emit(get_controller().replay(round_id, seed=seed))


@why_app.command("this")
def why_this(round_id: Annotated[int, typer.Argument()] = 0) -> None:
    controller = get_controller()
    round_id = round_id or controller.load_runtime_state().round_count
    emit(controller.why_this(round_id))


@why_app.command("not")
def why_not(action: Annotated[str, typer.Argument()], round_id: Annotated[int, typer.Argument()] = 0) -> None:
    controller = get_controller()
    round_id = round_id or controller.load_runtime_state().round_count
    emit(controller.why_not(round_id, action))


@what_app.command("changed")
def what_changed(window: int = 5) -> None:
    emit(get_controller().what_changed(window=window))


@eval_app.command("longrun")
def eval_longrun(rounds: int = 1000) -> None:
    emit(get_controller().eval_longrun(rounds=rounds))


@app.command("rest")
def rest_alias() -> None:
    emit(get_cil().relation_alias_rest())


@app.command("calm")
def calm_alias() -> None:
    emit(get_cil().relation_alias_calm())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
