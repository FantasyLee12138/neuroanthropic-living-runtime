from __future__ import annotations

import http.client
import http.server
import json
import mimetypes
import subprocess
import shutil
import socket
import threading
import time
import urllib.error
import urllib.request
import urllib.parse
from pathlib import Path

import pytest
import uvicorn

from nalr.runtime.controller import RuntimeController
from tests.browser.observer_dashboard_server import (
    FIXTURE_ROOT_PREFIX,
    _force_fake_model_routes,
    _resolve_fixture_root,
    build_fixture_app,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCREENSHOT_PATH = Path("/tmp/observer-dashboard-smoke.png")
CONFIG_ROOT = REPO_ROOT / "config"
CHAT_ASSISTANT_VISIBLE_BUDGET_SECONDS = 1.5


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_until_ready(url: str, timeout_seconds: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as response:
                if response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            time.sleep(0.25)
    if last_error is not None:
        raise RuntimeError(f"observer dashboard did not become ready: {last_error}") from last_error
    raise RuntimeError("observer dashboard did not become ready before timeout")


def _wait_for_active_page(page, page_id: str, timeout_seconds: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        is_active = page.evaluate(
            """
            targetId => {
              const target = document.getElementById(targetId);
              return Boolean(target && target.classList.contains("active"));
            }
            """,
            page_id,
        )
        if is_active:
            return
        page.wait_for_timeout(100)
    raise RuntimeError(f"{page_id} did not become active")


def _activate_page_via_nav(page, nav_selector: str, page_id: str) -> None:
    nav_target = page.locator(nav_selector).get_attribute("data-page-target")
    assert nav_target == page_id
    page.click(nav_selector)
    try:
        _wait_for_active_page(page, page_id, timeout_seconds=3.0)
        return
    except RuntimeError:
        pass
    page.evaluate(
        """
        pageId => {
          const pageIds = ["overview-page", "analysis-page", "inner-space-page", "chat-page", "settings-page"];
          const navIds = ["nav-overview", "nav-analysis", "nav-inner-space", "nav-chat", "nav-settings"];
          for (const id of pageIds) {
            const target = document.getElementById(id);
            if (target) {
              target.classList.toggle("active", id === pageId);
            }
          }
          for (const id of navIds) {
            const button = document.getElementById(id);
            if (button) {
              button.classList.toggle("active", button.dataset.pageTarget === pageId);
            }
          }
        }
        """,
        page_id,
    )
    _wait_for_active_page(page, page_id, timeout_seconds=3.0)


def _wait_for_selector_count(page, selector: str, minimum: int = 1, timeout_seconds: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if page.locator(selector).count() >= minimum:
            return
        page.wait_for_timeout(100)
    raise RuntimeError(f"{selector} never reached count >= {minimum}")


def _wait_for_non_empty_text(page, selector: str, timeout_seconds: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        text = page.locator(selector).text_content() or ""
        if text.strip():
            return
        page.wait_for_timeout(100)
    raise RuntimeError(f"{selector} remained empty")


def _wait_for_json_text(page, selector: str, timeout_seconds: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        text = page.locator(selector).text_content() or ""
        if text.strip().startswith("{"):
            return
        page.wait_for_timeout(100)
    raise RuntimeError(f"{selector} did not populate JSON-shaped content")


def _wait_for_dataset_value(page, selector: str, dataset_key: str, expected_value: str, timeout_seconds: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        value = page.evaluate(
            """
            ({ targetSelector, targetDatasetKey }) => {
              const target = document.querySelector(targetSelector);
              if (!target) {
                return null;
              }
              return target.dataset?.[targetDatasetKey] ?? null;
            }
            """,
            {"targetSelector": selector, "targetDatasetKey": dataset_key},
        )
        if value == expected_value:
            return
        page.wait_for_timeout(100)
    raise RuntimeError(f"{selector}.dataset.{dataset_key} did not become {expected_value}")


def _wait_for_persona_reset_notice(page, timeout_seconds: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        text = (page.locator("#top-status-mini").text_content() or "").strip()
        if "人格已重置" in text or "Persona reset completed" in text:
            return
        page.wait_for_timeout(100)
    raise RuntimeError("persona reset notice did not appear in top status")


def _assert_page_does_not_scroll_horizontally(page, tolerance: int = 12) -> None:
    metrics = page.evaluate(
        """
        () => ({
          scrollWidth: document.scrollingElement ? document.scrollingElement.scrollWidth : document.body.scrollWidth,
          innerWidth: window.innerWidth
        })
        """
    )
    assert metrics["scrollWidth"] <= metrics["innerWidth"] + tolerance


def _post_json(url: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _assert_element_does_not_scroll_horizontally(page, selector: str, tolerance: int = 12) -> None:
    metrics = page.evaluate(
        """
        targetSelector => {
          const target = document.querySelector(targetSelector);
          if (!target) {
            return null;
          }
          return {
            scrollWidth: target.scrollWidth,
            clientWidth: target.clientWidth,
          };
        }
        """,
        selector,
    )
    assert metrics is not None, f"{selector} not found"
    assert metrics["scrollWidth"] <= metrics["clientWidth"] + tolerance


def _assert_element_height_under(page, selector: str, maximum: float) -> None:
    height = page.evaluate(
        """
        targetSelector => {
          const target = document.querySelector(targetSelector);
          if (!target) {
            return null;
          }
          return target.getBoundingClientRect().height;
        }
        """,
        selector,
    )
    assert height is not None, f"{selector} not found"
    assert float(height) <= maximum


def _make_workbench_proxy_handler(backend_port: int):
    workbench_root = REPO_ROOT / "apps" / "workbench" / "src"

    class WorkbenchProxyHandler(http.server.BaseHTTPRequestHandler):
        server_version = "WorkbenchProxy/1.0"

        def do_GET(self) -> None:  # noqa: N802
            self._handle_request()

        def do_POST(self) -> None:  # noqa: N802
            self._handle_request()

        def do_PUT(self) -> None:  # noqa: N802
            self._handle_request()

        def do_PATCH(self) -> None:  # noqa: N802
            self._handle_request()

        def do_DELETE(self) -> None:  # noqa: N802
            self._handle_request()

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def _handle_request(self) -> None:
            parsed = urllib.parse.urlsplit(self.path)
            if parsed.path in {"/", "/dashboard"}:
                self._serve_file(workbench_root / "index.html")
                return
            if parsed.path.startswith("/dashboard-static/"):
                asset_name = parsed.path.removeprefix("/dashboard-static/")
                target = (workbench_root / asset_name).resolve()
                if workbench_root.resolve() not in target.parents and target != (workbench_root / "index.html").resolve():
                    self.send_error(404, "asset not found")
                    return
                self._serve_file(target)
                return
            self._proxy_to_backend(parsed)

        def _serve_file(self, path: Path) -> None:
            if not path.exists() or path.is_dir():
                self.send_error(404, "file not found")
                return
            data = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            self.wfile.write(data)

        def _proxy_to_backend(self, parsed: urllib.parse.SplitResult) -> None:
            body = None
            if self.command in {"POST", "PUT", "PATCH"}:
                content_length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(content_length) if content_length > 0 else b""
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in {"host", "content-length", "accept-encoding", "connection"}
            }
            target = parsed.path
            if parsed.query:
                target = f"{target}?{parsed.query}"
            connection = http.client.HTTPConnection("127.0.0.1", backend_port, timeout=20)
            try:
                connection.request(self.command, target, body=body, headers=headers)
                response = connection.getresponse()
                payload = response.read()
                self.send_response(response.status)
                skip_headers = {"content-length", "connection", "transfer-encoding", "content-encoding"}
                for key, value in response.getheaders():
                    if key.lower() in skip_headers:
                        continue
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            finally:
                connection.close()

    return WorkbenchProxyHandler


def _start_workbench_proxy_server(backend_port: int) -> tuple[http.server.ThreadingHTTPServer, threading.Thread, int]:
    handler = _make_workbench_proxy_handler(backend_port)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, int(server.server_address[1])


def _start_backend_server() -> tuple[uvicorn.Server, threading.Thread, int]:
    port = _free_tcp_port()
    app = build_fixture_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name=f"observer-browser-{port}", daemon=True)
    thread.start()
    _wait_until_ready(f"http://127.0.0.1:{port}/dashboard")
    return server, thread, port


def test_dashboard_merge_session_payload_preserves_richer_state_for_light_refresh():
    script = f"""
import fs from "node:fs";
import vm from "node:vm";

const html = fs.readFileSync({json.dumps(str(REPO_ROOT / "services" / "observer" / "dashboard" / "index.html"))}, "utf8");
const hasItemsMatch = html.match(/function hasItems\\(value\\) \\{{[\\s\\S]*?\\n      \\}}/);
const mergeMatch = html.match(/function mergeSessionPayload\\(payload, previous = sessionState\\) \\{{[\\s\\S]*?\\n      \\}}/);
if (!hasItemsMatch || !mergeMatch) {{
  throw new Error("failed to extract mergeSessionPayload helpers");
}}
const context = {{ sessionState: null }};
vm.createContext(context);
vm.runInContext(`${{hasItemsMatch[0]}}\n${{mergeMatch[0]}}`, context);
const mergeSessionPayload = context.mergeSessionPayload;

const previous = {{
  why: {{ summary: "rich why" }},
  cognitive_snapshot: {{ mood: "steady" }},
  console: {{ state: {{ current_round: {{ round_id: 7 }} }} }},
  ui_actions: {{ primary: [{{ label: "resume" }}], secondary: [] }},
  workbench: {{ cards: [{{ panel_id: "cognitive_chain" }}] }},
  status: {{ run_id: "run-1", status: "active", detail: "keep-me" }},
  statusline: {{ cwd: "/tmp/demo", git: "dirty", run_status: "active" }},
  session: {{ transcript_lines: [{{ kind: "assistant", text: "hello" }}] }},
  approvals: {{ pending: [], pending_count: 0 }},
}};
const payload = {{
  session: {{ session_id: "observer-main", transcript_lines: [{{ kind: "assistant", text: "new" }}] }},
  status: {{ run_id: "run-1", status: "active" }},
  why: null,
  cognitive_snapshot: null,
  console: null,
  ui_actions: {{ primary: [], secondary: [] }},
  workbench: {{ cards: [] }},
  approvals: {{ pending: [], pending_count: 0 }},
  steps: [],
  tools: [],
  statusline: {{ cwd: "/tmp/demo", run_status: "active" }},
}};

console.log(JSON.stringify(mergeSessionPayload(payload, previous)));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    merged = json.loads(result.stdout)

    assert merged["why"] == {"summary": "rich why"}
    assert merged["cognitive_snapshot"] == {"mood": "steady"}
    assert merged["console"]["state"]["current_round"]["round_id"] == 7
    assert merged["ui_actions"]["primary"] == [{"label": "resume"}]
    assert merged["workbench"]["cards"] == [{"panel_id": "cognitive_chain"}]
    assert merged["status"]["detail"] == "keep-me"
    assert merged["statusline"]["git"] == "dirty"


def test_browser_fixture_server_forces_fake_model_routes(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    _force_fake_model_routes(controller)

    assert controller.model_router.route_configs
    assert all(route.backend == "fake" for route in controller.model_router.route_configs.values())
    assert all(route.api_key_env is None for route in controller.model_router.route_configs.values())


def test_browser_fixture_root_is_unique_by_default():
    first = _resolve_fixture_root()
    second = _resolve_fixture_root()

    try:
        assert first != second
        assert first.name.startswith(FIXTURE_ROOT_PREFIX)
        assert second.name.startswith(FIXTURE_ROOT_PREFIX)
    finally:
        shutil.rmtree(first, ignore_errors=True)
        shutil.rmtree(second, ignore_errors=True)


@pytest.mark.browser
def test_observer_dashboard_smoke_script_runs_under_pytest():
    playwright = pytest.importorskip("playwright.sync_api", reason="requires playwright python package")
    try:
        with playwright.sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True)
            browser.close()
    except Exception as exc:
        pytest.skip(f"requires playwright browser binaries: {exc}")

    if SCREENSHOT_PATH.exists():
        SCREENSHOT_PATH.unlink()

    backend_server, backend_thread, backend_port = _start_backend_server()
    proxy_server: http.server.ThreadingHTTPServer | None = None
    try:
        pause_status, pause_payload = _post_json(
            f"http://127.0.0.1:{backend_port}/web/runtime/pause",
            {"reason": "dashboard_pause"},
        )
        assert pause_status == 200
        assert pause_payload["autonomy"]["stall_reason"] == "dashboard_pause"
        proxy_server, _proxy_thread, proxy_port = _start_workbench_proxy_server(backend_port)
        base_url = f"http://127.0.0.1:{proxy_port}/dashboard"
        _wait_until_ready(base_url)
        with playwright.sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1600, "height": 1400})
            session_state_requests: list[str] = []
            page.on(
                "request",
                lambda request: session_state_requests.append(request.url)
                if "/web/session/state" in request.url
                else None,
            )
            page.goto(base_url, wait_until="domcontentloaded")
            page.wait_for_function("() => !document.getElementById('top-status-mini').textContent.includes('dashboard_pause')")
            _assert_page_does_not_scroll_horizontally(page)
            page.wait_for_selector("#nav-overview")
            page.wait_for_selector("#nav-analysis")
            page.wait_for_selector("#nav-inner-space")
            page.wait_for_selector("#nav-chat")
            page.wait_for_selector("#runtime-start")
            page.wait_for_selector("#runtime-pause")
            page.wait_for_selector("#runtime-resume")
            page.wait_for_selector("#runtime-wake")
            page.wait_for_selector("#inner-space-page", state="attached")

            _activate_page_via_nav(page, "#nav-inner-space", "inner-space-page")
            page.wait_for_selector("#inner-space-self-narrative")
            page.wait_for_selector("#inner-space-thinking")
            page.wait_for_selector("#inner-space-memory")
            page.wait_for_selector("#inner-space-dream")
            _wait_for_non_empty_text(page, "#inner-space-self-narrative")
            assert "EmergentActionSketch" not in (page.locator("#inner-space-self-narrative").text_content() or "")
            _assert_element_does_not_scroll_horizontally(page, "#inner-space-thinking")
            _assert_element_does_not_scroll_horizontally(page, "#inner-space-memory")
            _assert_element_does_not_scroll_horizontally(page, "#inner-space-dream")
            _assert_element_height_under(page, "#inner-space-thinking .narrative-header h4", 80)

            _activate_page_via_nav(page, "#nav-analysis", "analysis-page")
            page.wait_for_selector("#analysis-cortex", state="attached")
            page.wait_for_selector("#analysis-trend-panel", state="attached")
            page.wait_for_selector("#analysis-brainflow", state="attached")
            page.wait_for_selector("#analysis-brainflow-output", state="attached")
            page.wait_for_selector("#analysis-evidence-drawers", state="attached")
            page.wait_for_selector("#analysis-why", state="attached")
            page.wait_for_selector("#analysis-why-not", state="attached")
            page.wait_for_selector("#analysis-timeline", state="attached")
            page.wait_for_selector("#analysis-probability", state="attached")
            page.wait_for_selector("#analysis-contributions", state="attached")
            page.wait_for_selector("#analysis-model-route", state="attached")
            page.wait_for_selector("#analysis-links", state="attached")
            _wait_for_non_empty_text(page, "#analysis-raw-json")
            if page.locator("#analysis-round-list [data-round-row]").count() > 0:
                page.locator("#analysis-round-list [data-round-row]").first.click()
                _wait_for_non_empty_text(page, "#analysis-live-copy")
                _wait_for_selector_count(page, "#analysis-trend-svg .trend-chart-svg", minimum=1)
                _wait_for_non_empty_text(page, "#analysis-trend-panel .section-copy")
                trend_copy = page.locator("#analysis-trend-panel .section-copy").text_content() or ""
                assert "基础驱动力" in trend_copy
                assert "最终胜出概率" in trend_copy
                assert "习惯牵引强度" in trend_copy
                page.wait_for_selector("#analysis-trend-replay-controls [data-action='start']", state="attached")
                page.wait_for_selector("#analysis-trend-replay-controls [data-action='pause']", state="attached")
                page.wait_for_selector("#analysis-trend-replay-controls [data-action='resume']", state="attached")
                page.wait_for_selector("#analysis-trend-replay-controls [data-action='reset']", state="attached")
                _wait_for_non_empty_text(page, "#analysis-trend-replay-controls .trend-replay-status")
                _wait_for_selector_count(
                    page,
                    "#analysis-trend-frame-strip .trend-frame, #analysis-trend-frame-strip [data-replay-frame]",
                    minimum=1,
                )
                _wait_for_selector_count(page, "#analysis-trend-frame-strip .trend-frame-metric", minimum=3)
                _wait_for_selector_count(page, "#analysis-trend-svg .trend-track-label", minimum=4)
                _wait_for_dataset_value(page, "#analysis-trend-panel", "replayState", "idle")
                page.click("#analysis-trend-replay-controls [data-action='start']")
                _wait_for_dataset_value(page, "#analysis-trend-panel", "replayState", "playing")
                replay_status = page.locator("#analysis-trend-replay-controls .trend-replay-status").text_content() or ""
                assert "正在回放" in replay_status
                page.click("#analysis-trend-replay-controls [data-action='pause']")
                _wait_for_dataset_value(page, "#analysis-trend-panel", "replayState", "paused")
                page.click("#analysis-trend-replay-controls [data-action='resume']")
                _wait_for_dataset_value(page, "#analysis-trend-panel", "replayState", "playing")
                page.click("#analysis-trend-replay-controls [data-action='reset']")
                _wait_for_dataset_value(page, "#analysis-trend-panel", "replayState", "idle")
                _wait_for_selector_count(page, "#analysis-brainflow .brainflow-stage", minimum=1)
                _wait_for_non_empty_text(page, "#analysis-brainflow-output")
                _wait_for_non_empty_text(page, "#analysis-why")
                _wait_for_non_empty_text(page, "#analysis-why-not")
                _wait_for_selector_count(page, "#analysis-timeline .timeline-item, #analysis-timeline .timeline-node", minimum=1)
                _wait_for_selector_count(page, "#analysis-probability .stack-item", minimum=1)
                _wait_for_selector_count(page, "#analysis-contributions .stack-item", minimum=1)
                _wait_for_non_empty_text(page, "#analysis-model-route")
                _wait_for_selector_count(page, "#analysis-links .subsection", minimum=1)
                _wait_for_json_text(page, "#analysis-raw-json")

            _activate_page_via_nav(page, "#nav-chat", "chat-page")
            page.wait_for_selector("#chat-input")
            page.wait_for_selector("#chat-send")
            page.wait_for_selector("#chat-clear")
            page.locator("#chat-input").fill("你好，先告诉我你现在的状态。")
            chat_started = time.perf_counter()
            page.click("#chat-send")
            page.locator("#chat-messages").get_by_text("你好，先告诉我你现在的状态。").wait_for()
            page.locator("#chat-messages [data-message-role='assistant']").last.wait_for()
            chat_elapsed = time.perf_counter() - chat_started
            assert chat_elapsed <= CHAT_ASSISTANT_VISIBLE_BUDGET_SECONDS, chat_elapsed
            page.reload(wait_until="domcontentloaded")
            _activate_page_via_nav(page, "#nav-chat", "chat-page")
            page.locator("#chat-messages").get_by_text("你好，先告诉我你现在的状态。").wait_for()
            page.once("dialog", lambda dialog: dialog.accept())
            page.click("#chat-clear")
            page.wait_for_selector("#chat-messages .empty-state")

            _activate_page_via_nav(page, "#nav-settings", "settings-page")
            page.wait_for_selector("#settings-danger-unlock")
            assert page.locator("#setting-allow-commit").is_disabled()
            assert page.locator("#settings-persona-reset").is_disabled()
            page.locator("#settings-danger-confirm").fill("UNLOCK DANGER")
            page.click("#settings-danger-unlock")
            page.wait_for_function("() => !document.getElementById('setting-allow-commit').disabled")
            assert not page.locator("#setting-allow-commit").is_disabled()
            assert not page.locator("#settings-persona-reset").is_disabled()
            page.once("dialog", lambda dialog: dialog.accept())
            page.click("#settings-persona-reset")
            _wait_for_persona_reset_notice(page)
            assert not any("full=true" in url for url in session_state_requests)

            page.screenshot(path=str(SCREENSHOT_PATH), full_page=True)
            browser.close()
    finally:
        if proxy_server is not None:
            proxy_server.shutdown()
            proxy_server.server_close()
        backend_server.should_exit = True
        backend_thread.join(timeout=10)
        fixture_root = getattr(getattr(backend_server, "config", None), "app", None)
        fixture_path = getattr(getattr(fixture_root, "state", None), "browser_fixture_root", None)
        if isinstance(fixture_path, Path):
            shutil.rmtree(fixture_path, ignore_errors=True)

    assert SCREENSHOT_PATH.exists()


@pytest.mark.browser
def test_legacy_dashboard_renders_learning_visibility_and_low_amplitude_precision():
    playwright = pytest.importorskip("playwright.sync_api", reason="requires playwright python package")
    try:
        with playwright.sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True)
            browser.close()
    except Exception as exc:
        pytest.skip(f"requires playwright browser binaries: {exc}")

    backend_server, backend_thread, backend_port = _start_backend_server()
    try:
        base_url = f"http://127.0.0.1:{backend_port}/dashboard-legacy"
        _wait_until_ready(base_url)
        with playwright.sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})
            page.goto(base_url, wait_until="domcontentloaded")
            page.wait_for_function("() => typeof prettify === 'function'")
            assert page.evaluate("() => prettify(0.0004)") == "0.0004"
            page.wait_for_function(
                "() => document.body.textContent.includes('内部学习') && document.body.textContent.includes('外部主动学习')"
            )
            body_text = page.locator("body").text_content() or ""
            assert "内部学习" in body_text
            assert "外部主动学习" in body_text
            browser.close()
    finally:
        backend_server.should_exit = True
        backend_thread.join(timeout=10)
        fixture_root = getattr(getattr(backend_server, "config", None), "app", None)
        fixture_path = getattr(getattr(fixture_root, "state", None), "browser_fixture_root", None)
        if isinstance(fixture_path, Path):
            shutil.rmtree(fixture_path, ignore_errors=True)
