from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from nalr.cil.runtime import CommandInterfaceLayer
from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent, to_dict


app = typer.Typer(help="NeuroAnthropic Living Runtime CLI")
state_app = typer.Typer()
body_app = typer.Typer()
mood_app = typer.Typer()
focus_app = typer.Typer()
memory_app = typer.Typer()
habit_app = typer.Typer()
mode_app = typer.Typer()
relation_app = typer.Typer()
trace_app = typer.Typer()
agent_app = typer.Typer()
skill_app = typer.Typer()
debug_app = typer.Typer()
nudge_app = typer.Typer()
replay_app = typer.Typer()
suppress_app = typer.Typer()
checkpoint_app = typer.Typer()
safe_app = typer.Typer()
budget_app = typer.Typer()
explain_app = typer.Typer()

app.add_typer(state_app, name="state")
app.add_typer(body_app, name="body")
app.add_typer(mood_app, name="mood")
app.add_typer(focus_app, name="focus")
app.add_typer(relation_app, name="relation")
app.add_typer(memory_app, name="memory")
app.add_typer(habit_app, name="habit")
app.add_typer(mode_app, name="mode")
app.add_typer(trace_app, name="trace")
app.add_typer(agent_app, name="agent")
app.add_typer(skill_app, name="skill")
app.add_typer(debug_app, name="debug")
app.add_typer(nudge_app, name="nudge")
app.add_typer(replay_app, name="replay")
app.add_typer(suppress_app, name="suppress")
app.add_typer(checkpoint_app, name="checkpoint")
app.add_typer(safe_app, name="safe")
app.add_typer(budget_app, name="budget")
app.add_typer(explain_app, name="explain")


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
    emit(get_cil().execute("state show"))


@body_app.command("show")
def body_show() -> None:
    emit(get_cil().execute("body show"))


@mood_app.command("show")
def mood_show() -> None:
    emit(get_cil().execute("mood show"))


@focus_app.command("show")
def focus_show() -> None:
    controller = get_controller()
    if controller.load_runtime_state().round_count == 0:
        controller.tick(RoundEvent(source="system", content="focus probe"), scenario=os.environ.get("NALR_SCENARIO", "chat"), mode=os.environ.get("NALR_MODE", "interactive"))
    emit(get_cil().execute("focus show"))


@relation_app.command("show")
def relation_show(target: str) -> None:
    emit(get_cil().execute(f"relation show {target}"))


@memory_app.command("top")
def memory_top(limit: int = 5) -> None:
    emit(get_controller().memory_top(limit))


@habit_app.command("top")
def habit_top(limit: int = 5) -> None:
    emit(get_controller().habit_top(limit))


@mode_app.command("set")
def mode_set(mode_name: str) -> None:
    emit(get_cil().execute(f"mode set {mode_name}"))


@trace_app.command("round")
def trace_round(round_id: int) -> None:
    emit(get_cil().execute(f"trace round {round_id}"))


@trace_app.command("why")
def trace_why(round_id: int) -> None:
    emit(get_cil().execute(f"trace why {round_id}"))


@trace_app.command("contribution")
def trace_contribution(round_id: int) -> None:
    emit(get_cil().execute(f"trace contribution {round_id}"))


@agent_app.command("list")
def agent_list() -> None:
    emit(get_cil().execute("agent list"))


@agent_app.command("disable")
def agent_disable(agent_name: str) -> None:
    emit(get_cil().execute(f"agent disable {agent_name}"))


@agent_app.command("enable")
def agent_enable(agent_name: str) -> None:
    emit(get_cil().execute(f"agent enable {agent_name}"))


@skill_app.command("stats")
def skill_stats() -> None:
    emit(get_cil().execute("skill stats"))


@skill_app.command("profile")
def skill_profile(skill_name: str) -> None:
    emit(get_cil().execute(f"skill profile {skill_name}"))


@debug_app.command("weight")
def debug_weight(agent_name: str, weight: float) -> None:
    emit(get_cil().execute(f"debug weight {agent_name} {weight}"))


@nudge_app.command("focus")
def nudge_focus(delta: float) -> None:
    emit(get_cil().execute(f"nudge focus {delta}"))


@replay_app.command("round")
def replay_round(round_id: int, seed: int | None = None) -> None:
    suffix = f" {seed}" if seed is not None else ""
    emit(get_cil().execute(f"replay round {round_id}{suffix}"))


@suppress_app.command("dmn")
def suppress_dmn(ttl: str = "temporary") -> None:
    emit(get_cil().execute(f"suppress dmn {ttl}"))


@checkpoint_app.command("create")
def checkpoint_create() -> None:
    emit(get_cil().execute("checkpoint create"))


@checkpoint_app.command("rewind")
def checkpoint_rewind(checkpoint_id: str) -> None:
    emit(get_cil().execute(f"checkpoint rewind {checkpoint_id}"))


@safe_app.command("on")
def safe_on() -> None:
    emit(get_cil().execute("safe on"))


@safe_app.command("off")
def safe_off() -> None:
    emit(get_cil().execute("safe off"))


@budget_app.command("show")
def budget_show() -> None:
    emit(get_cil().execute("budget show"))


@explain_app.command("current")
def explain_current() -> None:
    emit(get_cil().execute("explain current"))


@app.command("rest")
def rest() -> None:
    emit(get_cil().execute("rest"))


@app.command("calm")
def calm() -> None:
    emit(get_cil().execute("calm"))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
