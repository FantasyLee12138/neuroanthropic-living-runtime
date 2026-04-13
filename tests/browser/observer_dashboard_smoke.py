from __future__ import annotations

import os
from pathlib import Path

from playwright.sync_api import sync_playwright


BASE_URL = f"http://127.0.0.1:{int(os.environ.get('NALR_OBSERVER_BROWSER_PORT', '8765') or '8765')}/dashboard"
SCREENSHOT_PATH = Path("/tmp/observer-dashboard-smoke.png")


def assert_page_does_not_scroll(page, tolerance: int = 12) -> None:
    metrics = page.evaluate(
        """
        () => ({
          scrollHeight: document.scrollingElement ? document.scrollingElement.scrollHeight : document.body.scrollHeight,
          innerHeight: window.innerHeight
        })
        """
    )
    if metrics["scrollHeight"] > metrics["innerHeight"] + tolerance:
        raise RuntimeError(f"page scroll height overflow: {metrics}")


def wait_for_active_page(page, page_id: str, timeout_ms: int = 20_000) -> None:
    page.wait_for_function(
        """
        targetId => {
          const target = document.getElementById(targetId);
          return Boolean(target && target.classList.contains("active"));
        }
        """,
        page_id,
        timeout=timeout_ms,
    )


def wait_for_selector_count(page, selector: str, minimum: int = 1, timeout_ms: int = 20_000) -> None:
    page.wait_for_function(
        """
        ({ selector, minimum }) => document.querySelectorAll(selector).length >= minimum
        """,
        {"selector": selector, "minimum": minimum},
        timeout=timeout_ms,
    )


def wait_for_non_empty_text(page, selector: str, timeout_ms: int = 20_000) -> None:
    page.wait_for_function(
        """
        targetSelector => {
          const target = document.querySelector(targetSelector);
          return Boolean(target && target.textContent && target.textContent.trim().length > 0);
        }
        """,
        selector,
        timeout=timeout_ms,
    )


def wait_for_json_text(page, selector: str, timeout_ms: int = 20_000) -> None:
    page.wait_for_function(
        """
        targetSelector => {
          const target = document.querySelector(targetSelector);
          if (!target || !target.textContent) {
            return false;
          }
          return target.textContent.trim().startsWith("{");
        }
        """,
        selector,
        timeout=timeout_ms,
    )


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1400})
        session_state_requests: list[str] = []
        page.on(
            "request",
            lambda request: session_state_requests.append(request.url)
            if "/web/session/state" in request.url
            else None,
        )
        page.goto(BASE_URL, wait_until="networkidle")
        page.wait_for_timeout(1500)
        assert_page_does_not_scroll(page)
        page.wait_for_selector("#overview-page.active")
        page.wait_for_selector("#nav-overview")
        page.wait_for_selector("#nav-analysis")
        page.wait_for_selector("#nav-chat")
        page.wait_for_selector("#runtime-start")
        page.wait_for_selector("#runtime-pause")
        page.wait_for_selector("#runtime-resume")
        page.wait_for_selector("#runtime-wake")
        page.wait_for_selector("#overview-jump-analysis")
        wait_for_selector_count(page, "#autonomy-summary .stack-item", minimum=1)
        wait_for_selector_count(page, "#model-summary .stack-item", minimum=1)
        wait_for_non_empty_text(page, "#learning-mode")
        wait_for_selector_count(page, "#overview-rounds .list-row", minimum=1)

        page.click("#overview-jump-analysis")
        wait_for_active_page(page, "analysis-page")
        wait_for_selector_count(page, "#analysis-round-list [data-round-row]", minimum=1)
        page.locator("#analysis-round-list [data-round-row]").first.click()
        wait_for_non_empty_text(page, "#analysis-live-copy")
        wait_for_selector_count(page, "#analysis-trend-svg .trend-chart-svg", minimum=1)
        wait_for_selector_count(page, "#analysis-brainflow .brainflow-stage", minimum=1)
        wait_for_non_empty_text(page, "#analysis-brainflow-output")
        wait_for_non_empty_text(page, "#analysis-why")
        wait_for_non_empty_text(page, "#analysis-why-not")
        wait_for_selector_count(page, "#analysis-timeline .timeline-node, #analysis-timeline .detail-row", minimum=1)
        wait_for_selector_count(page, "#analysis-probability .stack-item", minimum=1)
        wait_for_selector_count(page, "#analysis-contributions .stack-item", minimum=1)
        wait_for_non_empty_text(page, "#analysis-model-route")
        wait_for_selector_count(page, "#analysis-links .subsection", minimum=1)
        wait_for_json_text(page, "#analysis-raw-json")

        page.click("#nav-chat")
        wait_for_active_page(page, "chat-page")
        page.wait_for_selector("#chat-input")
        page.wait_for_selector("#chat-send")
        page.locator("#chat-input").fill("你好，先告诉我你现在的状态。")
        page.click("#chat-send")
        page.wait_for_selector("#chat-messages [data-message-role='user'] .message-body")
        page.locator("#chat-messages").get_by_text("你好，先告诉我你现在的状态。").wait_for()
        page.wait_for_selector("#chat-messages [data-message-role='assistant'] .message-body")
        if any("full=true" in url for url in session_state_requests):
            raise RuntimeError(f"dashboard requested full session state: {session_state_requests}")

        page.screenshot(path=str(SCREENSHOT_PATH), full_page=True)
        browser.close()


if __name__ == "__main__":
    main()
