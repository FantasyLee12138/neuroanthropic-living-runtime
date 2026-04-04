from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from nalr.cli.presentation import (
    format_agents_view,
    format_chat_turn,
    format_gates_view,
    format_skills_view,
    format_why_view,
)
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
identity_app = typer.Typer()
run_app = typer.Typer()
trace_app = typer.Typer()
trace_export_app = typer.Typer()
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
dream_app = typer.Typer()
why_app = typer.Typer()
what_app = typer.Typer()
eval_app = typer.Typer()

app.add_typer(state_app, name="state")
app.add_typer(body_app, name="body")
app.add_typer(mood_app, name="mood")
app.add_typer(focus_app, name="focus")
app.add_typer(identity_app, name="identity")
app.add_typer(run_app, name="run")
app.add_typer(relation_app, name="relation")
app.add_typer(memory_app, name="memory")
app.add_typer(habit_app, name="habit")
app.add_typer(mode_app, name="mode")
app.add_typer(trace_app, name="trace")
trace_app.add_typer(trace_export_app, name="export")
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
app.add_typer(dream_app, name="dream")
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


def default_scenario() -> str:
    return os.environ.get("NALR_SCENARIO", "chat")


def default_mode() -> str:
    return os.environ.get("NALR_MODE", "interactive")


def build_chat_payload(result) -> dict[str, object]:
    return {
        "round_id": result.round_id,
        "sampled_action": to_dict(result.sampled_action),
        "rendered_expression": to_dict(result.rendered_expression),
        "top_drivers": to_dict(result.trace.top_drivers),
        "identity": to_dict(result.state.identity_state),
    }


def resolve_show_sections(show: str | None) -> list[str]:
    if not show:
        return []
    normalized = show.strip().lower()
    if normalized == "all":
        return ["why", "agents", "skills", "gates"]
    if normalized not in {"why", "agents", "skills", "gates"}:
        raise typer.BadParameter("show must be one of: why, agents, skills, gates, all")
    return [normalized]


def attach_trace_sections(controller: RuntimeController, payload: dict[str, object], sections: list[str]) -> dict[str, object]:
    round_id = payload["round_id"]
    if "why" in sections:
        payload["why"] = controller.why_this(round_id)
    if "agents" in sections:
        payload["agents"] = controller.trace_agents(round_id)
    if "skills" in sections:
        payload["skills"] = controller.trace_skills(round_id)
    if "gates" in sections:
        payload["gates"] = controller.trace_gates(round_id)
    return payload


def render_section(section: str, payload: dict[str, object]) -> str:
    if section == "why":
        return format_why_view(payload["why"])
    if section == "agents":
        return format_agents_view(payload["agents"])
    if section == "skills":
        return format_skills_view(payload["skills"])
    if section == "gates":
        return format_gates_view(payload["gates"])
    raise ValueError(f"unsupported section: {section}")


def emit_chat_output(payload: dict[str, object], sections: list[str], *, as_json: bool) -> None:
    if as_json:
        emit(payload)
        return
    chunks = [format_chat_turn(payload)]
    for section in sections:
        chunks.append(render_section(section, payload))
    typer.echo("\n\n".join(chunks))


def emit_trace_view(
    *,
    round_ref: str,
    as_json: bool,
    fetcher,
    formatter,
) -> None:
    try:
        payload = fetcher(round_ref)
    except FileNotFoundError:
        typer.echo("No rounds yet. Send a message first.")
        raise typer.Exit(code=1)
    if as_json:
        emit(payload)
        return
    typer.echo(formatter(payload))


def emit_json_trace(round_ref: str, fetcher) -> None:
    try:
        emit(fetcher(round_ref))
    except FileNotFoundError:
        typer.echo("No rounds yet. Send a message first.")
        raise typer.Exit(code=1)


def run_chat_turn(
    *,
    controller: RuntimeController,
    message: str,
    target: str | None,
    cue: str | None,
    scenario: str,
    mode: str,
    valence: float,
    energy_delta: float,
    name: str | None = None,
):
    if name:
        controller.seed_identity_name(name, source_hint="user_seed")
    return controller.tick(
        RoundEvent(
            source="user",
            content=message,
            target=target,
            cue=cue,
            valence=valence,
            energy_delta=energy_delta,
        ),
        scenario=scenario,
        mode=mode,
    )


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


@identity_app.command("show")
def identity_show() -> None:
    emit(get_cil().execute("identity show"))


@identity_app.command("set-name")
def identity_set_name(name: str) -> None:
    emit(get_cil().execute(f"identity set-name {name}"))


@run_app.command("start")
def run_start(goal: list[str] = typer.Argument(..., help="goal for the autonomous run")) -> None:
    emit(get_cil().execute(f"run start {' '.join(goal).strip()}"))


@run_app.command("status")
def run_status() -> None:
    emit(get_cil().execute("run status"))


@run_app.command("pause")
def run_pause() -> None:
    emit(get_cil().execute("run pause"))


@run_app.command("resume")
def run_resume() -> None:
    emit(get_cil().execute("run resume"))


@run_app.command("abort")
def run_abort() -> None:
    emit(get_cil().execute("run abort"))


@run_app.command("explain")
def run_explain() -> None:
    emit(get_cil().execute("run explain"))


@relation_app.command("show")
def relation_show(target: str) -> None:
    emit(get_cil().execute(f"relation show {target}"))


@memory_app.command("top")
def memory_top(limit: int = 5) -> None:
    emit(get_controller().memory_top(limit))


@memory_app.command("recall")
def memory_recall(cue: str) -> None:
    emit(get_cil().execute(f"memory recall {cue}"))


@memory_app.command("compact")
def memory_compact() -> None:
    emit(get_controller().compact_memory())


@memory_app.command("sample")
def memory_sample(
    tier: str = typer.Option(..., "--tier"),
    limit: int = typer.Option(5, "--limit"),
    cue: str | None = typer.Option(None, "--cue"),
) -> None:
    emit(get_controller().sample_memory(tier, limit=limit, cue=cue))


@habit_app.command("top")
def habit_top(limit: int = 5) -> None:
    emit(get_controller().habit_top(limit))


@habit_app.command("reset")
def habit_reset(pattern: str) -> None:
    emit(get_cil().execute(f"habit reset {pattern}"))


@mode_app.command("set")
def mode_set(mode_name: str) -> None:
    emit(get_cil().execute(f"mode set {mode_name}"))


@trace_app.command("round")
def trace_round(round_ref: str) -> None:
    emit_json_trace(round_ref, get_controller().trace_round)


@trace_app.command("why")
def trace_why(round_ref: str) -> None:
    emit_json_trace(round_ref, get_controller().why_this)


@trace_app.command("contribution")
def trace_contribution(round_ref: str) -> None:
    emit_json_trace(round_ref, get_controller().contribution_breakdown)


@trace_app.command("agents")
def trace_agents(
    round_ref: str,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    controller = get_controller()
    emit_trace_view(
        round_ref=round_ref,
        as_json=json_output,
        fetcher=controller.trace_agents,
        formatter=format_agents_view,
    )


@trace_app.command("skills")
def trace_skills(
    round_ref: str,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    controller = get_controller()
    emit_trace_view(
        round_ref=round_ref,
        as_json=json_output,
        fetcher=controller.trace_skills,
        formatter=format_skills_view,
    )


@trace_app.command("gates")
def trace_gates(
    round_ref: str,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    controller = get_controller()
    emit_trace_view(
        round_ref=round_ref,
        as_json=json_output,
        fetcher=controller.trace_gates,
        formatter=format_gates_view,
    )


@trace_export_app.command("parquet")
def trace_export_parquet(
    since_round: int | None = typer.Option(None, "--since-round"),
    overwrite: bool = typer.Option(True, "--overwrite/--no-overwrite"),
) -> None:
    emit(get_controller().export_trace_parquet(since_round=since_round, overwrite=overwrite))


@trace_app.command("compact")
def trace_compact() -> None:
    emit(get_cil().execute("trace compact"))


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


@nudge_app.command("relation")
def nudge_relation(target: str, metric: str, delta: float) -> None:
    emit(get_cil().execute(f"nudge relation {target} {metric} {delta:+.2f}"))


@replay_app.command("round")
def replay_round(round_id: int, seed: int | None = None) -> None:
    suffix = f" {seed}" if seed is not None else ""
    emit(get_cil().execute(f"replay round {round_id}{suffix}"))


@why_app.command("this")
def why_this(round_id: Annotated[int, typer.Argument()] = 0) -> None:
    suffix = f" {round_id}" if round_id else ""
    emit(get_cil().execute(f"why this{suffix}"))


@why_app.command("not")
def why_not(action: Annotated[str, typer.Argument()], round_id: Annotated[int, typer.Argument()] = 0) -> None:
    suffix = f" {round_id}" if round_id else ""
    emit(get_cil().execute(f"why not {action}{suffix}"))


@what_app.command("changed")
def what_changed(window: Annotated[int, typer.Argument()] = 5) -> None:
    emit(get_cil().execute(f"what changed {window}"))


@eval_app.command("longrun")
def eval_longrun(rounds: Annotated[int, typer.Argument()] = 1000) -> None:
    emit(get_cil().execute(f"eval longrun {rounds}"))


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


@budget_app.command("set")
def budget_set(cap: int = typer.Option(..., "--cap")) -> None:
    emit(get_cil().execute(f"budget set --cap {cap}"))


@explain_app.command("current")
def explain_current() -> None:
    emit(get_cil().execute("explain current"))


@dream_app.command("status")
def dream_status() -> None:
    emit(get_controller().dream_status())


@dream_app.command("trace")
def dream_trace(round_ref: str = typer.Argument("last")) -> None:
    try:
        emit(get_controller().dream_trace(round_ref))
    except FileNotFoundError:
        typer.echo("No dream runs yet.")
        raise typer.Exit(code=1)


@dream_app.command("proposals")
def dream_proposals(run_ref: str = typer.Argument("last")) -> None:
    try:
        emit(get_controller().dream_proposals(run_ref))
    except FileNotFoundError:
        typer.echo("No dream runs yet.")
        raise typer.Exit(code=1)


@dream_app.command("metrics")
def dream_metrics() -> None:
    emit(get_controller().dream_metrics())


@dream_app.command("run")
def dream_run(
    mode: str = typer.Option("sleep", "--mode"),
    cue: str | None = typer.Option(None, "--cue"),
) -> None:
    emit(get_controller().run_dream(mode=mode, cue=cue))


@dream_app.command("on")
def dream_on() -> None:
    emit(get_controller().set_dream_enabled(True))


@dream_app.command("off")
def dream_off() -> None:
    emit(get_controller().set_dream_enabled(False))


@app.command("Dream")
def dream_alias(cue: str | None = typer.Argument(None)) -> None:
    emit(get_controller().run_dream(mode="sleep", cue=cue))


@app.command("rest")
def rest() -> None:
    emit(get_cil().execute("rest"))


@app.command("calm")
def calm() -> None:
    emit(get_cil().execute("calm"))


@app.command("chat")
def chat(
    message: list[str] = typer.Argument(..., help="message to send to the runtime"),
    target: str | None = typer.Option(None, "--target"),
    cue: str | None = typer.Option(None, "--cue"),
    name: str | None = typer.Option(None, "--name"),
    scenario: str | None = typer.Option(None, "--scenario"),
    mode: str | None = typer.Option(None, "--mode"),
    valence: float = typer.Option(0.0, "--valence"),
    energy_delta: float = typer.Option(0.0, "--energy-delta"),
    show: str | None = typer.Option(None, "--show"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    controller = get_controller()
    effective_scenario = scenario or default_scenario()
    effective_mode = mode or default_mode()
    result = run_chat_turn(
        controller=controller,
        message=" ".join(message).strip(),
        target=target,
        cue=cue,
        scenario=effective_scenario,
        mode=effective_mode,
        valence=valence,
        energy_delta=energy_delta,
        name=name,
    )
    payload = build_chat_payload(result)
    sections = resolve_show_sections(show)
    attach_trace_sections(controller, payload, sections)
    emit_chat_output(payload, sections, as_json=json_output)


@app.command("repl")
def repl(
    target: str | None = typer.Option(None, "--target"),
    name: str | None = typer.Option(None, "--name"),
    scenario: str | None = typer.Option(None, "--scenario"),
    mode: str | None = typer.Option(None, "--mode"),
) -> None:
    controller = get_controller()
    cil = CommandInterfaceLayer(controller)
    effective_scenario = scenario or default_scenario()
    current_mode = mode or default_mode()
    last_round_id: int | None = None
    if name:
        controller.seed_identity_name(name, source_hint="user_seed")

    typer.echo("Entering NALR REPL. Type /help for commands, /exit to quit.")

    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            typer.echo("")
            break
        if not line:
            continue
        if line.startswith("/"):
            parts = line[1:].split()
            command = parts[0].lower() if parts else ""
            args = parts[1:]

            if command in {"exit", "quit"}:
                typer.echo("Bye.")
                break
            if command == "help":
                typer.echo(
                    "\n".join(
                        [
                            "Commands:",
                            "/help",
                            "/run <goal>",
                            "/status",
                            "/pause",
                            "/resume",
                            "/abort",
                            "/checkpoint",
                            "/why",
                            "/agents",
                            "/skills",
                            "/gates",
                            "/state",
                            "/mode <name>",
                            "/safe on|off",
                            "/budget",
                            "/exit",
                        ]
                    )
                )
                continue
            if command == "run":
                if not args:
                    typer.echo("Usage: /run <goal>")
                    continue
                emit(cil.execute(f"run start {' '.join(args)}"))
                continue
            if command == "status":
                emit(cil.execute("run status"))
                continue
            if command == "pause":
                emit(cil.execute("run pause"))
                continue
            if command == "resume":
                emit(cil.execute("run resume"))
                continue
            if command == "abort":
                emit(cil.execute("run abort"))
                continue
            if command == "checkpoint":
                emit(cil.execute("checkpoint create"))
                continue
            if command in {"why", "agents", "skills", "gates"}:
                if last_round_id is None:
                    typer.echo("No rounds yet. Send a message first.")
                    continue
                if command == "why":
                    typer.echo(format_why_view(controller.why_this(last_round_id)))
                elif command == "agents":
                    typer.echo(format_agents_view(controller.trace_agents(last_round_id)))
                elif command == "skills":
                    typer.echo(format_skills_view(controller.trace_skills(last_round_id)))
                else:
                    typer.echo(format_gates_view(controller.trace_gates(last_round_id)))
                continue
            if command == "state":
                emit(cil.execute("state show"))
                continue
            if command == "budget":
                emit(cil.execute("budget show"))
                continue
            if command == "mode":
                if len(args) != 1:
                    typer.echo("Usage: /mode <name>")
                    continue
                current_mode = args[0]
                emit(cil.execute(f"mode set {current_mode}"))
                continue
            if command == "safe":
                if len(args) != 1 or args[0] not in {"on", "off"}:
                    typer.echo("Usage: /safe on|off")
                    continue
                emit(cil.execute(f"safe {args[0]}"))
                continue

            typer.echo("Unknown command. Type /help for available commands.")
            continue

        result = run_chat_turn(
            controller=controller,
            message=line,
            target=target,
            cue=None,
            scenario=effective_scenario,
            mode=current_mode,
            valence=0.0,
            energy_delta=0.0,
            name=None,
        )
        last_round_id = result.round_id
        typer.echo(format_chat_turn(build_chat_payload(result)))
        typer.echo("")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
