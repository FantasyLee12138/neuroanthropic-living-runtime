from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from nalr.runtime.controller import RuntimeController


def create_app(project_root: Path | str | None = None, config_root: Path | str | None = None) -> FastAPI:
    project_root_path = Path(project_root) if project_root else Path.cwd()
    effective_config_root = Path(config_root) if config_root else Path(os.environ.get("NALR_CONFIG_DIR", project_root_path / "config"))
    controller = RuntimeController(project_root=project_root_path, config_root=effective_config_root)
    app = FastAPI(title="NALR Observer", version="0.1.0")

    @app.get("/state")
    def state() -> dict:
        return controller.state_payload()

    @app.get("/trace/{round_id}")
    def trace(round_id: int) -> dict:
        try:
            return controller.trace_round(round_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/why/{round_id}")
    def why(round_id: int) -> dict:
        try:
            return controller.why_this(round_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/contributions/{round_id}")
    def contributions(round_id: int) -> dict:
        try:
            return controller.contribution_breakdown(round_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/memory/top")
    def memory_top(limit: int = 5) -> list[dict]:
        return controller.memory_top(limit=limit)

    @app.get("/habit/top")
    def habit_top(limit: int = 5) -> list[dict]:
        return controller.habit_top(limit=limit)

    @app.get("/metrics/summary")
    def metrics_summary() -> dict:
        return controller.metrics_summary()

    dashboard_path = project_root_path / "services" / "observer" / "dashboard" / "index.html"
    if not dashboard_path.exists():
        dashboard_path = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"

    @app.get("/dashboard")
    def dashboard() -> FileResponse:
        return FileResponse(dashboard_path)

    return app


app = create_app()
