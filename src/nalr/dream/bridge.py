from __future__ import annotations

import json
import sys
from typing import TextIO

from nalr.dream.engine import OneiroiAgent
from nalr.schemas.models import DreamRunRequest


def serve_stdio(stdin: TextIO, stdout: TextIO) -> None:
    agent = OneiroiAgent()
    for raw_line in stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            if payload.get("type") != "dream.start":
                raise ValueError("unsupported dream message type")
            request = DreamRunRequest(**payload["request"])
            result = agent.run(request)
            stdout.write(json.dumps({"type": "dream.result", "result": result}, ensure_ascii=False) + "\n")
            stdout.flush()
        except Exception as exc:
            stdout.write(json.dumps({"type": "dream.error", "message": str(exc)}, ensure_ascii=False) + "\n")
            stdout.flush()


def main() -> None:
    serve_stdio(sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
