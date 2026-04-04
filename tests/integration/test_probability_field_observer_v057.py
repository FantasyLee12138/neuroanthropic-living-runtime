from __future__ import annotations

import inspect
from pathlib import Path

from fastapi.testclient import TestClient

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent
from services.observer.api.app import create_app


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_observer_exposes_probability_field_routes(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Plan the next step and remember tea.",
            target="user",
            cue="tea",
        ),
        scenario="task",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    app = create_app(project_root=tmp_path, config_root=CONFIG_ROOT)
    route_map = {route.path: route.endpoint for route in app.routes if hasattr(route, "path")}

    assert "/metrics/probability-field" in route_map, "expected a probability-field summary route"
    assert "/trace/{round_id}/probability-field" in route_map, "expected a probability-field trace route"
    assert inspect.iscoroutinefunction(route_map["/metrics/probability-field"])
    assert inspect.iscoroutinefunction(route_map["/trace/{round_id}/probability-field"])


def test_observer_probability_field_routes_return_layered_contribution_data(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Please plan dinner and remember rice.",
            target="user",
            cue="rice",
        ),
        scenario="task",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    summary_response = client.get("/metrics/probability-field")
    trace_response = client.get("/trace/1/probability-field")

    assert summary_response.status_code == 200
    assert trace_response.status_code == 200
    summary_payload = summary_response.json()
    trace_payload = trace_response.json()
    assert "layers" in summary_payload
    assert "module_heatmap" in summary_payload
    assert "winner_posterior" in trace_payload
    assert "counterfactual_top_peaks" in trace_payload
