from __future__ import annotations

import json
import os
from pathlib import Path

import typer

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
trace_app = typer.Typer()
agent_app = typer.Typer()
checkpoint_app = typer.Typer()
safe_app = typer.Typer()
budget_app = typer.Typer()

app.add_typer(state_app, name="state")
app.add_typer(body_app, name="body")
app.add_typer(mood_app, name="mood")
app.add_typer(focus_app, name="focus")
app.add_typer(memory_app, name="memory")
app.add_typer(habit_app, name="habit")
app.add_typer(mode_app, name="mode")
app.add_typer(trace_app, name="trace")
app.add_typer(agent_app, name="agent")
app.add_typer(checkpoint_app, name="checkpoint")
app.add_typer(safe_app, name="safe")
app.add_typer(budget_app, name="budget")


def get_controller() -> RuntimeController:
    home = Path(os.environ.get("NALR_HOME", ".alive"))
    config_dir = Path(os.environ.get("NALR_CONFIG_DIR", "config"))
    project_root = home.parent if home.name == ".alive" else Path.cwd()
    return RuntimeController(project_root=project_root, config_root=config_dir, home_path=home)


def emit(payload: object) -> None:
    typer.echo(json.dumps(to_dict(payload), ensure_ascii=False, indent=2))


@state_app.command("show")
def state_show() -> None:
    emit(get_controller().state_payload())


@body_app.command("show")
def body_show() -> None:
    emit({"body_energy": get_controller().load_runtime_state().body_energy})


@mood_app.command("show")
def mood_show() -> None:
    emit({"mood": get_controller().load_runtime_state().mood})


@focus_app.command("show")
def focus_show() -> None:
    controller = get_controller()
    if controller.load_runtime_state().round_count == 0:
        controller.tick(RoundEvent(source="system", content="focus probe"), scenario=os.environ.get("NALR_SCENARIO", "chat"), mode=os.environ.get("NALR_MODE", "interactive"))
    emit({"focus": controller.load_runtime_state().focus})


@memory_app.command("top")
def memory_top(limit: int = 5) -> None:
    emit(get_controller().memory_top(limit))


@habit_app.command("top")
def habit_top(limit: int = 5) -> None:
    emit(get_controller().habit_top(limit))


@mode_app.command("set")
def mode_set(mode_name: str) -> None:
    emit(get_controller().apply_command(f"mode set {mode_name}"))


@trace_app.command("round")
def trace_round(round_id: int) -> None:
    emit(get_controller().trace_round(round_id))


@agent_app.command("list")
def agent_list() -> None:
    emit(get_controller().agent_list())


@agent_app.command("disable")
def agent_disable(agent_name: str) -> None:
    emit(get_controller().apply_command(f"agent disable {agent_name}"))


@agent_app.command("enable")
def agent_enable(agent_name: str) -> None:
    emit(get_controller().apply_command(f"agent enable {agent_name}"))


@checkpoint_app.command("create")
def checkpoint_create() -> None:
    emit(get_controller().checkpoint())


@safe_app.command("on")
def safe_on() -> None:
    emit(get_controller().apply_command("safe on"))


@safe_app.command("off")
def safe_off() -> None:
    emit(get_controller().apply_command("safe off"))


@budget_app.command("show")
def budget_show() -> None:
    emit({"budget_remaining": get_controller().load_runtime_state().budget_remaining})


def main() -> None:
    app()


if __name__ == "__main__":
    main()

