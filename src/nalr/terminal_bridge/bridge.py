from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TextIO

from nalr.runtime.controller import RuntimeController
from nalr.terminal_bridge.handlers import TerminalEventHandler
from nalr.terminal_bridge.protocol import ProtocolError, build_outbound_event


def _build_controller() -> RuntimeController:
    home = Path(os.environ.get("NALR_HOME", ".alive"))
    config_dir = Path(os.environ.get("NALR_CONFIG_DIR", "config"))
    project_root = home.parent if home.name == ".alive" else Path.cwd()
    return RuntimeController(project_root=project_root, config_root=config_dir, home_path=home)


def serve_stdio(stdin: TextIO, stdout: TextIO) -> None:
    handler = TerminalEventHandler(_build_controller())
    for raw_line in stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            events = handler.handle(payload)
        except (json.JSONDecodeError, ProtocolError, FileNotFoundError, ValueError) as exc:
            events = [build_outbound_event("error", message=str(exc))]
        for event in events:
            stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
            stdout.flush()


def main() -> None:
    serve_stdio(sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
