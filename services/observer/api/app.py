from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from nalr.runtime.controller import RuntimeController


def create_app(project_root: Path | str, config_root: Path | str | None = None) -> FastAPI:
    controller = RuntimeController(project_root=Path(project_root), config_root=Path(config_root) if config_root else None)
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

    @app.get("/memory/top")
    def memory_top(limit: int = 5) -> list[dict]:
        return controller.memory_top(limit=limit)

    @app.get("/habit/top")
    def habit_top(limit: int = 5) -> list[dict]:
        return controller.habit_top(limit=limit)

    return app
