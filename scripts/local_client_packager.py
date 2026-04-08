#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import plistlib
import shutil
import stat
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APP_NAME = "NALR Local Client.app"
DEFAULT_BUNDLE_IDENTIFIER = "com.nalr.local-client"
DEFAULT_BUNDLE_VERSION = "1.5.0"
APP_PAYLOAD_DIRNAME = "app"
EMBEDDED_RUNTIME_DIRNAME = ".venv"
LAUNCH_AGENT_STATE_DIR = "${HOME}/.nalr/local-client/observer-service"
COPYTREE_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")


def _launcher_command() -> Path:
    return REPO_ROOT / "Open_NALR_Workbench.command"


def _launcher_observer() -> Path:
    return REPO_ROOT / "alive-observer"


def _payload_sources() -> tuple[tuple[str, Path], ...]:
    return (
        ("src", REPO_ROOT / "src"),
        ("services", REPO_ROOT / "services"),
        ("config", REPO_ROOT / "config"),
    )


def _default_runtime_source() -> Path:
    return REPO_ROOT / EMBEDDED_RUNTIME_DIRNAME


def _copy_executable(source: Path, destination: Path) -> None:
    shutil.copy2(source, destination)
    destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, ignore=COPYTREE_IGNORE)


def _write_wrapper(destination: Path) -> None:
    wrapper = """#!/bin/zsh
set -euo pipefail

APP_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$APP_ROOT/Resources/Open_NALR_Workbench.command" "$@"
"""
    destination.write_text(wrapper, encoding="utf-8")
    destination.chmod(0o755)


def _write_observer_launcher(destination: Path) -> None:
    launcher = f"""#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
APP_ROOT="$SCRIPT_DIR/{APP_PAYLOAD_DIRNAME}"
EMBEDDED_RUNTIME="$APP_ROOT/{EMBEDDED_RUNTIME_DIRNAME}"

load_env() {{
  candidate="$1"
  [ -f "$candidate" ] || return 1
  set -a
  . "$candidate"
  set +a
  return 0
}}

load_env "$SCRIPT_DIR/.env.local" || \
  load_env "$SCRIPT_DIR/.env" || \
  load_env "$APP_ROOT/.env.local" || \
  load_env "$APP_ROOT/.env" || \
  true

supports_observer_service() {{
  candidate="$1"
  [ -x "$candidate" ] || return 1
  "$candidate" -c "import fastapi, uvicorn" >/dev/null 2>&1
}}

if [ "${{NALR_OBSERVER_PYTHON_BIN:-}}" != "" ] && [ -x "${{NALR_OBSERVER_PYTHON_BIN}}" ]; then
  PYTHON_BIN="${{NALR_OBSERVER_PYTHON_BIN}}"
elif supports_observer_service "$EMBEDDED_RUNTIME/bin/python"; then
  PYTHON_BIN="$EMBEDDED_RUNTIME/bin/python"
elif supports_observer_service "$EMBEDDED_RUNTIME/bin/python3.11"; then
  PYTHON_BIN="$EMBEDDED_RUNTIME/bin/python3.11"
elif supports_observer_service "/opt/homebrew/opt/python@3.11/bin/python3.11"; then
  PYTHON_BIN="/opt/homebrew/opt/python@3.11/bin/python3.11"
else
  PYTHON_BIN="${{PYTHON:-python3}}"
fi

if [ "${{PYTHONPATH:-}}" = "" ]; then
  export PYTHONPATH="$APP_ROOT:$APP_ROOT/src"
else
  export PYTHONPATH="$APP_ROOT:$APP_ROOT/src:$PYTHONPATH"
fi

export NALR_LOCAL_CLIENT_BUNDLE_ROOT="$SCRIPT_DIR"
export NALR_PROJECT_ROOT="${{NALR_PROJECT_ROOT:-$APP_ROOT}}"
export NALR_CONFIG_DIR="${{NALR_CONFIG_DIR:-$APP_ROOT/config}}"
export NALR_OBSERVER_SERVICE_DIR="${{NALR_OBSERVER_SERVICE_DIR:-{LAUNCH_AGENT_STATE_DIR}}}"

mkdir -p "$NALR_OBSERVER_SERVICE_DIR"
cd "$APP_ROOT"
exec "$PYTHON_BIN" -m nalr.observer_launcher "$@"
"""
    destination.write_text(launcher, encoding="utf-8")
    destination.chmod(0o755)


def _write_bundle_manifest(destination: Path, *, runtime_embedded: bool) -> None:
    payload = {
        "bundle_payload_root": f"Contents/Resources/{APP_PAYLOAD_DIRNAME}",
        "embedded_runtime": runtime_embedded,
        "runtime_path": (
            f"Contents/Resources/{APP_PAYLOAD_DIRNAME}/{EMBEDDED_RUNTIME_DIRNAME}"
            if runtime_embedded
            else None
        ),
        "launcher_scripts": [
            "Contents/Resources/Open_NALR_Workbench.command",
            "Contents/Resources/alive-observer",
        ],
        "payload_roots": [
            f"Contents/Resources/{APP_PAYLOAD_DIRNAME}/src",
            f"Contents/Resources/{APP_PAYLOAD_DIRNAME}/services",
            f"Contents/Resources/{APP_PAYLOAD_DIRNAME}/config",
        ],
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_info_plist(destination: Path, *, app_name: str) -> None:
    payload = {
        "CFBundleDevelopmentRegion": "en",
        "CFBundleDisplayName": app_name.removesuffix(".app"),
        "CFBundleExecutable": "Open NALR Workbench",
        "CFBundleIdentifier": DEFAULT_BUNDLE_IDENTIFIER,
        "CFBundleName": app_name.removesuffix(".app"),
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": DEFAULT_BUNDLE_VERSION,
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "12.0",
    }
    with destination.open("wb") as handle:
        plistlib.dump(payload, handle, fmt=plistlib.FMT_XML)


def _resolve_runtime_source(runtime_source: Path | None) -> Path | None:
    candidate = runtime_source.expanduser().resolve() if runtime_source is not None else _default_runtime_source()
    if runtime_source is None and not candidate.exists():
        return None
    if not candidate.exists():
        raise FileNotFoundError(f"missing runtime source: {candidate}")
    if not candidate.is_dir():
        raise ValueError(f"runtime source must be a directory: {candidate}")
    python_candidates = [
        candidate / "bin" / "python",
        candidate / "bin" / "python3",
        candidate / "bin" / "python3.11",
    ]
    if not any(path.exists() and path.is_file() for path in python_candidates):
        if runtime_source is None:
            return None
        raise FileNotFoundError(f"runtime source does not contain a reusable Python executable: {candidate}")
    return candidate


def build_local_client_app(
    output_dir: Path,
    *,
    app_name: str = DEFAULT_APP_NAME,
    runtime_source: Path | None = None,
) -> Path:
    launcher_command = _launcher_command()
    launcher_observer = _launcher_observer()
    if not launcher_command.exists():
        raise FileNotFoundError(f"missing launcher command: {launcher_command}")
    if not launcher_observer.exists():
        raise FileNotFoundError(f"missing launcher executable: {launcher_observer}")
    for name, source in _payload_sources():
        if not source.exists():
            raise FileNotFoundError(f"missing bundle payload source {name}: {source}")

    resolved_runtime_source = _resolve_runtime_source(runtime_source)
    output_dir = output_dir.expanduser().resolve()
    bundle_dir = output_dir / app_name
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)

    contents_dir = bundle_dir / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"
    payload_root = resources_dir / APP_PAYLOAD_DIRNAME
    macos_dir.mkdir(parents=True, exist_ok=True)
    resources_dir.mkdir(parents=True, exist_ok=True)
    payload_root.mkdir(parents=True, exist_ok=True)

    for relative_name, source in _payload_sources():
        _copy_tree(source, payload_root / relative_name)
    if resolved_runtime_source is not None:
        _copy_tree(resolved_runtime_source, payload_root / EMBEDDED_RUNTIME_DIRNAME)

    _write_wrapper(macos_dir / "Open NALR Workbench")
    _copy_executable(launcher_command, resources_dir / launcher_command.name)
    _write_observer_launcher(resources_dir / launcher_observer.name)
    _write_bundle_manifest(resources_dir / "bundle-manifest.json", runtime_embedded=resolved_runtime_source is not None)
    _write_info_plist(contents_dir / "Info.plist", app_name=app_name)

    return bundle_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize the NALR local-client macOS app bundle.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="dist",
        help="Directory where the .app bundle will be created.",
    )
    parser.add_argument(
        "--app-name",
        default=DEFAULT_APP_NAME,
        help="Bundle name to create.",
    )
    parser.add_argument(
        "--runtime-source",
        help="Optional reusable runtime/venv directory to embed into the bundle. Defaults to repo .venv when reusable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    bundle_dir = build_local_client_app(
        Path(args.output_dir),
        app_name=args.app_name,
        runtime_source=Path(args.runtime_source) if args.runtime_source else None,
    )
    print(bundle_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
