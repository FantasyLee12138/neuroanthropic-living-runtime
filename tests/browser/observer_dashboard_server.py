from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import uvicorn


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "config"
DEFAULT_PORT = 8765
FIXTURE_ROOT_PREFIX = "nalr_observer_browser_fixture_"
_AUTO_FIXTURE_ROOTS: set[Path] = set()
for path in (REPO_ROOT, REPO_ROOT / "src"):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from nalr.providers.router import FakeBackend, ModelRouteConfig, ModelRouter
from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent


def _cleanup_auto_fixture_roots() -> None:
    for fixture_root in list(_AUTO_FIXTURE_ROOTS):
        if fixture_root.exists():
            shutil.rmtree(fixture_root, ignore_errors=True)
        _AUTO_FIXTURE_ROOTS.discard(fixture_root)


atexit.register(_cleanup_auto_fixture_roots)


def _resolve_fixture_root() -> Path:
    configured_root = str(os.environ.get("NALR_OBSERVER_BROWSER_FIXTURE_ROOT") or "").strip()
    if configured_root:
        return Path(configured_root)
    fixture_root = Path(tempfile.mkdtemp(prefix=FIXTURE_ROOT_PREFIX))
    _AUTO_FIXTURE_ROOTS.add(fixture_root)
    return fixture_root


def _force_fake_model_routes(controller: RuntimeController) -> None:
    for route in controller.model_router.route_configs.values():
        route.backend = "fake"
        route.api_key_env = None


def _install_deterministic_model_stubs() -> None:
    fake_backend = FakeBackend()

    def _route_config(router: ModelRouter, route_name: str) -> ModelRouteConfig:
        route = router.route_configs.get(route_name)
        if route is not None:
            route.backend = "fake"
            route.api_key_env = None
            return route
        return ModelRouteConfig(
            name=route_name,
            backend="fake",
            model="observer-browser-stub",
            timeout_ms=1,
            retries=0,
            enabled=True,
        )

    def _generate(self: ModelRouter, route_name: str, request):
        route = _route_config(self, str(route_name))
        return fake_backend.generate(route=route, request=request)

    def _generate_config(self: ModelRouter, route: ModelRouteConfig, request):
        return fake_backend.generate(route=route, request=request)

    def _stream_generate(self: ModelRouter, route_name: str, request):
        response = _generate(self, route_name, request)
        text = str(response.payload.get("text") or response.raw_text or "").strip()
        if text:
            yield text

    def _stream_generate_config(self: ModelRouter, route: ModelRouteConfig, request):
        response = _generate_config(self, route, request)
        text = str(response.payload.get("text") or response.raw_text or "").strip()
        if text:
            yield text

    ModelRouter.generate = _generate
    ModelRouter.generate_config = _generate_config
    ModelRouter.stream_generate = _stream_generate
    ModelRouter.stream_generate_config = _stream_generate_config


def seed_fixture(fixture_root: Path | None = None) -> Path:
    root = fixture_root or _resolve_fixture_root()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    controller = RuntimeController(project_root=root, config_root=CONFIG_ROOT)
    _force_fake_model_routes(controller)
    state = controller.load_runtime_state()
    state.body_energy = 0.11
    state.fatigue = 0.88
    state.memory_fragments = 0.79
    state.self_continuity = 0.31
    state.meaning_strength = 0.22
    state.subjective_state.felt = ["累", "想停下", "先往里收"]
    state.subjective_state.spontaneous = 0.77
    state.subjective_state.reject_all = 0.68
    state.subjective_state.boundary = 0.72
    state.subjective_state.meaning_made = ["先吸收内部线索，再决定如何表达"]
    controller._save_state(state, sync=True)

    rounds = [
        RoundEvent(source="user", content="先别急着答复，先看看内部状态。", target="user", cue="内部状态"),
        RoundEvent(source="user", content="如果不回应，会发生什么？", target="user", cue="不回应"),
        RoundEvent(source="user", content="把当前动作分布和四维空间一起给我看。", target="user", cue="动作分布"),
    ]
    for event in rounds:
        controller.tick(event, scenario="chat", mode="interactive")
    controller.flush_pending_io(raise_on_error=True)
    return root


def build_fixture_app():
    _install_deterministic_model_stubs()
    os.environ.setdefault("NALR_OBSERVER_SKIP_DEFAULT_APP", "1")
    os.environ.setdefault("NALR_OBSERVER_DISABLE_BACKGROUND_RUNNERS", "1")
    from services.observer.api.app import create_app

    fixture_root = seed_fixture(_resolve_fixture_root())
    app = create_app(project_root=fixture_root, config_root=CONFIG_ROOT)
    _force_fake_model_routes(app.state.controller)
    app.state.browser_fixture_root = fixture_root
    return app


def main() -> None:
    app = build_fixture_app()
    port = int(os.environ.get("NALR_OBSERVER_BROWSER_PORT", str(DEFAULT_PORT)) or DEFAULT_PORT)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
