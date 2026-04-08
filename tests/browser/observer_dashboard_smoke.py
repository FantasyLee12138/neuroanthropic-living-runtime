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


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1400})
        page.goto(BASE_URL, wait_until="networkidle")
        page.wait_for_timeout(1500)
        assert_page_does_not_scroll(page)
        page.wait_for_selector("#start-chat")
        page.wait_for_selector("#go-analysis")
        page.wait_for_selector("#go-replay")

        page.click("#start-chat")
        page.wait_for_selector("#chat-page.active")
        page.wait_for_selector("#chat-page .section-title")
        page.wait_for_selector("#chat-input")
        assert_page_does_not_scroll(page)
        page.wait_for_selector("#command-toggle")
        page.click("#command-toggle")
        page.wait_for_selector("#command-dialog:not(.hidden)")
        page.wait_for_selector("#command-input")
        page.click("#command-close")
        page.locator("#command-dialog").wait_for(state="hidden")
        page.click("#process-toggle")
        page.wait_for_selector("#process-dialog:not(.hidden)")
        page.locator("#process-dialog").get_by_role("heading", name="步骤流").wait_for()
        page.click("#process-close")
        page.locator("#process-dialog").wait_for(state="hidden")
        page.wait_for_timeout(3500)
        if "未连接" in (page.locator("#session-badge").text_content() or ""):
            raise RuntimeError("chat session was not initialized after entering the dashboard")
        page.locator("#chat-input").fill("你好，先告诉我你现在的状态。")
        page.click("#send-chat")
        page.wait_for_timeout(5000)
        page.locator("#messages").get_by_text("你好，先告诉我你现在的状态。").wait_for()
        page.wait_for_selector("#messages [data-message-role='assistant'] .message-body")

        page.locator("#go-analysis").click()
        page.wait_for_selector("#analysis-page.active")
        page.locator("#analysis-page [data-workbench-tab='instinct']").click()
        page.wait_for_selector("#workbench-panel")
        page.locator("#workbench-panel").get_by_text("当前落点").wait_for()
        page.locator("#workbench-panel").get_by_text("优胜区").wait_for()
        page.locator("#analysis-page").get_by_role("heading", name="四层概率场").wait_for()
        page.locator("#probability-layers").get_by_text("情境层").wait_for()
        page.locator("#probability-layers").get_by_text("峰值焦点").first.wait_for()
        page.locator("#probability-layers").get_by_text("Token").wait_for()

        page.locator("#go-replay").click()
        page.wait_for_selector("#replay-page.active")
        page.locator("#replay-page").get_by_text("反事实回放").wait_for()
        page.locator("#replay-page").get_by_text("未采纳路径").wait_for()

        page.screenshot(path=str(SCREENSHOT_PATH), full_page=True)
        browser.close()


if __name__ == "__main__":
    main()
