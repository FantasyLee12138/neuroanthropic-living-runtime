from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from nalr.runtime.controller import RuntimeController
from nalr.terminal_bridge.session import TerminalSessionStore


def create_app(project_root: Path | str | None = None, config_root: Path | str | None = None) -> FastAPI:
    project_root_path = Path(project_root) if project_root else Path.cwd()
    effective_config_root = Path(config_root) if config_root else Path(os.environ.get("NALR_CONFIG_DIR", project_root_path / "config"))
    controller = RuntimeController(project_root=project_root_path, config_root=effective_config_root)
    terminal_sessions = TerminalSessionStore(controller.runtime_dir)
    app = FastAPI(title="NALR Observer", version="0.1.0")

    @app.get("/state")
    async def state() -> dict:
        return controller.state_payload()

    @app.get("/identity")
    async def identity() -> dict:
        return controller.identity_payload()

    @app.get("/runs/current")
    async def current_run() -> dict:
        try:
            return controller.run_status()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/steps")
    async def run_steps(run_id: str) -> dict:
        try:
            return controller.run_steps(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/tools")
    async def run_tools(run_id: str) -> dict:
        try:
            return controller.run_tools(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/sessions/current")
    async def current_terminal_session() -> dict:
        try:
            payload = terminal_sessions.read_current().__dict__
            payload["runtime_session_id"] = controller.load_runtime_state().session_id
            return payload
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/sessions/{session_id}")
    async def terminal_session_detail(session_id: str) -> dict:
        try:
            return terminal_sessions.read(session_id).__dict__
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/trace/{round_id}")
    async def trace(round_id: int) -> dict:
        try:
            return controller.trace_round(round_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/why/{round_id}")
    async def why(round_id: int) -> dict:
        try:
            return controller.why_this(round_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/contributions/{round_id}")
    async def contributions(round_id: int) -> dict:
        try:
            return controller.contribution_breakdown(round_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/memory/top")
    async def memory_top(limit: int = 5) -> list[dict]:
        return controller.memory_top(limit=limit)

    @app.get("/memory/recall/{cue}")
    async def memory_recall(cue: str) -> dict:
        return controller.memory_recall(cue)

    @app.get("/habit/top")
    async def habit_top(limit: int = 5) -> list[dict]:
        return controller.habit_top(limit=limit)

    @app.get("/metrics/summary")
    async def metrics_summary() -> dict:
        return controller.metrics_summary()

    @app.get("/metrics/authenticity")
    async def authenticity_metrics() -> dict:
        return controller.authenticity_timeline()

    @app.get("/metrics/vitality")
    async def vitality_metrics() -> dict:
        return controller.vitality_timeline()

    @app.get("/dream/status")
    async def dream_status() -> dict:
        return controller.dream_status()

    @app.get("/dream/runs")
    async def dream_runs() -> dict:
        return controller.dream_runs()

    @app.get("/dream/runs/{round_ref}")
    async def dream_trace(round_ref: str) -> dict:
        try:
            return controller.dream_trace(round_ref)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/dream/metrics")
    async def dream_metrics() -> dict:
        return controller.dream_metrics()

    @app.get("/skills/stats")
    async def skill_stats() -> dict:
        return controller.skill_stats()

    @app.get("/skills/profile/{skill_name}")
    async def skill_profile(skill_name: str) -> dict:
        return controller.skill_profile(skill_name)

    @app.get("/metrics/conflicts")
    async def conflict_timeline() -> dict:
        return controller.conflict_timeline()

    @app.get("/metrics/entropy")
    async def entropy_metrics() -> dict:
        return controller.entropy_metrics()

    @app.get("/metrics/mode-switches")
    async def mode_switch_timeline() -> dict:
        return controller.mode_switch_timeline()

    @app.get("/analysis/ablation")
    async def ablation_summary() -> dict:
        return controller.ablation_summary()

    dashboard_path = project_root_path / "services" / "observer" / "dashboard" / "index.html"
    if not dashboard_path.exists():
        dashboard_path = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"

    @app.get("/dashboard")
    async def dashboard() -> FileResponse:
        return FileResponse(dashboard_path)

    return app


app = create_app()
