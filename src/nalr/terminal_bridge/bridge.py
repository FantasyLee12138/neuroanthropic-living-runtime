from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from collections.abc import Iterable
from typing import TextIO

from nalr.runtime.controller import RuntimeController
from nalr.terminal_bridge.handlers import TerminalEventHandler
from nalr.terminal_bridge.protocol import ProtocolError, build_outbound_event


STATUS_EVENT_TYPES = {"run_status", "step_update", "tool_call", "tool_result", "sidebar_snapshot"}


def _build_controller() -> RuntimeController:
    home = Path(os.environ.get("NALR_HOME", ".alive"))
    config_dir = Path(os.environ.get("NALR_CONFIG_DIR", "config"))
    project_root = home.parent if home.name == ".alive" else Path.cwd()
    return RuntimeController(project_root=project_root, config_root=config_dir, home_path=home)


def _write_events(stdout: TextIO, events: Iterable[dict]) -> None:
    buffered_lines: list[str] = []
    for event in events:
        encoded = json.dumps(event, ensure_ascii=False) + "\n"
        event_type = str(event.get("type") or "")
        if event_type == "assistant_token":
            if buffered_lines:
                stdout.write("".join(buffered_lines))
                stdout.flush()
                buffered_lines = []
            stdout.write(encoded)
            stdout.flush()
            continue
        if event_type in STATUS_EVENT_TYPES:
            buffered_lines.append(encoded)
            continue
        buffered_lines.append(encoded)
    if buffered_lines:
        stdout.write("".join(buffered_lines))
        stdout.flush()


def serve_stdio(stdin: TextIO, stdout: TextIO) -> None:
    handler = TerminalEventHandler(_build_controller())
    for raw_line in stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            events = handler.handle_stream(payload)
            _write_events(stdout, events)
        except (json.JSONDecodeError, ProtocolError, FileNotFoundError, ValueError) as exc:
            _write_events(stdout, [build_outbound_event("error", message=str(exc))])


def main() -> None:
    serve_stdio(sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
