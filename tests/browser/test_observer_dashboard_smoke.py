from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCREENSHOT_PATH = Path("/tmp/observer-dashboard-smoke.png")


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


def _wait_for_active_page(page, page_id: str) -> None:
    deadline = time.monotonic() + 20.0
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


def _wait_for_text_not_contains(page, selector: str, needle: str, timeout_seconds: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        text = page.locator(selector).text_content() or ""
        if needle not in text:
            return
        page.wait_for_timeout(100)
    raise RuntimeError(f"{selector} still contains {needle!r}")


def _assert_page_does_not_scroll(page, tolerance: int = 12) -> None:
    metrics = page.evaluate(
        """
        () => ({
          scrollHeight: document.scrollingElement ? document.scrollingElement.scrollHeight : document.body.scrollHeight,
          innerHeight: window.innerHeight
        })
        """
    )
    assert metrics["scrollHeight"] <= metrics["innerHeight"] + tolerance


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

    port = _free_tcp_port()
    base_url = f"http://127.0.0.1:{port}/dashboard"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT / 'src'}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else f"{REPO_ROOT}:{REPO_ROOT / 'src'}"
    env["NALR_OBSERVER_BROWSER_PORT"] = str(port)
    server = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "tests/browser/observer_dashboard_server.py")],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        _wait_until_ready(base_url)
        with playwright.sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1600, "height": 1400})
            page.goto(base_url, wait_until="networkidle")
            page.wait_for_timeout(1500)
            _assert_page_does_not_scroll(page)
            page.evaluate("document.querySelector('#start-chat').click()")
            _wait_for_active_page(page, "chat-page")
            page.wait_for_selector("#chat-input")
            _wait_for_text_not_contains(page, "#session-badge", "未连接")
            _assert_page_does_not_scroll(page)

            assert page.locator("#command-dialog").is_hidden()
            page.click("#command-toggle")
            page.wait_for_selector("#command-dialog:not(.hidden)")
            page.wait_for_selector("#command-input")
            page.click("#command-close")
            page.locator("#command-dialog").wait_for(state="hidden")

            assert page.locator("#process-dialog").is_hidden()
            page.click("#process-toggle")
            page.wait_for_selector("#process-dialog:not(.hidden)")
            page.locator("#process-dialog").get_by_role("heading", name="步骤流").wait_for()
            page.click("#process-close")
            page.locator("#process-dialog").wait_for(state="hidden")

            page.locator("#chat-input").fill("你好，先告诉我你现在的状态。")
            page.click("#send-chat")
            page.wait_for_selector("#messages [data-message-role='user'] .message-body")
            page.locator("#messages").get_by_text("你好，先告诉我你现在的状态。").wait_for()

            page.evaluate("document.querySelector('#nav-analysis-link').click()")
            _wait_for_active_page(page, "analysis-page")
            page.locator("#analysis-page [data-workbench-tab='instinct']").click()
            page.locator("#workbench-panel").get_by_text("当前落点").wait_for()
            page.locator("#workbench-panel").get_by_text("优胜区").first.wait_for()
            page.locator("#analysis-page").get_by_role("heading", name="四层概率场").wait_for()
            page.locator("#probability-layers").get_by_text("情境层", exact=True).wait_for()
            page.locator("#probability-layers").get_by_text("峰值焦点").first.wait_for()

            page.evaluate("document.querySelector('#nav-replay-link').click()")
            _wait_for_active_page(page, "replay-page")
            page.locator("#replay-page").get_by_text("反事实回放").first.wait_for()
            page.locator("#replay-page").get_by_text("未采纳路径").wait_for()

            page.screenshot(path=str(SCREENSHOT_PATH), full_page=True)
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)

    assert SCREENSHOT_PATH.exists()
