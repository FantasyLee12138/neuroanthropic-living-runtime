from __future__ import annotations

import importlib.util
import json
import plistlib
import stat
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGER_PATH = REPO_ROOT / "scripts" / "local_client_packager.py"


def load_packager_module():
    spec = importlib.util.spec_from_file_location("local_client_packager", PACKAGER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_file(path: Path, content: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def _sample_repo(tmp_path: Path, *, include_runtime: bool) -> Path:
    repo_root = tmp_path / "repo"
    _write_file(
        repo_root / "Open_NALR_Workbench.command",
        "#!/bin/zsh\nset -euo pipefail\nROOT_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\nexec \"$ROOT_DIR/alive-observer\" start \"$@\"\n",
        executable=True,
    )
    _write_file(
        repo_root / "alive-observer",
        "#!/usr/bin/env sh\nset -eu\nSCRIPT_DIR=$(CDPATH= cd -- \"$(dirname \"$0\")\" && pwd)\ncd \"$SCRIPT_DIR\"\nexec python3 -m nalr.observer_launcher \"$@\"\n",
        executable=True,
    )
    _write_file(repo_root / "src" / "nalr" / "__init__.py", "")
    _write_file(repo_root / "src" / "nalr" / "observer_launcher.py", "def main():\n    return 0\n")
    _write_file(repo_root / "services" / "__init__.py", "")
    _write_file(repo_root / "services" / "observer" / "__init__.py", "")
    _write_file(repo_root / "services" / "observer" / "api" / "__init__.py", "")
    _write_file(repo_root / "services" / "observer" / "api" / "app.py", "def create_app():\n    return object()\n")
    _write_file(repo_root / "services" / "observer" / "dashboard" / "index.html", "<html>dashboard</html>\n")
    _write_file(repo_root / "config" / "models.yaml", "models: []\n")
    _write_file(repo_root / "config" / "agents.yaml", "agents: []\n")
    if include_runtime:
        _write_file(repo_root / ".venv" / "bin" / "python", "#!/usr/bin/env sh\nexit 0\n", executable=True)
    return repo_root


@pytest.fixture
def packager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = load_packager_module()
    repo_root = _sample_repo(tmp_path, include_runtime=True)
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    return module


def test_build_local_client_app_materializes_bundle_payload_and_runtime(tmp_path: Path, packager) -> None:
    bundle_dir = packager.build_local_client_app(tmp_path / "dist")

    assert bundle_dir == (tmp_path / "dist" / "NALR Local Client.app")
    contents_dir = bundle_dir / "Contents"
    macos_wrapper = contents_dir / "MacOS" / "Open NALR Workbench"
    resources_command = contents_dir / "Resources" / "Open_NALR_Workbench.command"
    resources_observer = contents_dir / "Resources" / "alive-observer"
    payload_root = contents_dir / "Resources" / "app"
    manifest_path = contents_dir / "Resources" / "bundle-manifest.json"
    info_plist = contents_dir / "Info.plist"

    assert macos_wrapper.exists()
    assert resources_command.exists()
    assert resources_observer.exists()
    assert (payload_root / "src" / "nalr" / "observer_launcher.py").exists()
    assert (payload_root / "services" / "observer" / "api" / "app.py").exists()
    assert (payload_root / "services" / "observer" / "dashboard" / "index.html").exists()
    assert (payload_root / "config" / "models.yaml").exists()
    assert (payload_root / ".venv" / "bin" / "python").exists()
    assert manifest_path.exists()
    assert info_plist.exists()

    assert macos_wrapper.stat().st_mode & stat.S_IXUSR
    assert resources_command.stat().st_mode & stat.S_IXUSR
    assert resources_observer.stat().st_mode & stat.S_IXUSR

    wrapper_text = macos_wrapper.read_text(encoding="utf-8")
    assert "Resources/Open_NALR_Workbench.command" in wrapper_text

    command_text = resources_command.read_text(encoding="utf-8")
    assert command_text == (packager.REPO_ROOT / "Open_NALR_Workbench.command").read_text(encoding="utf-8")

    observer_text = resources_observer.read_text(encoding="utf-8")
    assert "APP_ROOT=\"$SCRIPT_DIR/app\"" in observer_text
    assert "EMBEDDED_RUNTIME=\"$APP_ROOT/.venv\"" in observer_text
    assert "PYTHONPATH=\"$APP_ROOT:$APP_ROOT/src" in observer_text
    assert "NALR_CONFIG_DIR" in observer_text
    assert "NALR_OBSERVER_SERVICE_DIR" in observer_text
    assert "cd \"$APP_ROOT\"" in observer_text
    assert "exec \"$PYTHON_BIN\" -m nalr.observer_launcher \"$@\"" in observer_text

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["bundle_payload_root"] == "Contents/Resources/app"
    assert manifest["embedded_runtime"] is True
    assert manifest["runtime_path"] == "Contents/Resources/app/.venv"

    with info_plist.open("rb") as handle:
        plist = plistlib.load(handle)

    assert plist["CFBundleExecutable"] == "Open NALR Workbench"
    assert plist["CFBundleIdentifier"] == "com.nalr.local-client"
    assert plist["CFBundlePackageType"] == "APPL"
    assert plist["CFBundleShortVersionString"] == "1.5.0"
    assert plist["CFBundleName"] == "NALR Local Client"


def test_build_local_client_app_rebuild_overwrites_previous_bundle(tmp_path: Path, packager) -> None:
    bundle_dir = packager.build_local_client_app(tmp_path / "dist")
    stale_file = bundle_dir / "Contents" / "Resources" / "stale.txt"
    stale_file.write_text("stale\n", encoding="utf-8")

    rebuilt_bundle = packager.build_local_client_app(tmp_path / "dist")

    assert rebuilt_bundle == bundle_dir
    assert not stale_file.exists()
    assert (bundle_dir / "Contents" / "Resources" / "app" / "config" / "agents.yaml").exists()
    assert (bundle_dir / "Contents" / "Resources" / "app" / ".venv" / "bin" / "python").exists()


def test_build_local_client_app_without_runtime_keeps_clear_bootstrap_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packager = load_packager_module()
    repo_root = _sample_repo(tmp_path, include_runtime=False)
    monkeypatch.setattr(packager, "REPO_ROOT", repo_root)

    bundle_dir = packager.build_local_client_app(tmp_path / "dist")
    payload_root = bundle_dir / "Contents" / "Resources" / "app"
    manifest_path = bundle_dir / "Contents" / "Resources" / "bundle-manifest.json"

    assert (payload_root / "src" / "nalr" / "observer_launcher.py").exists()
    assert (payload_root / "services" / "observer" / "api" / "app.py").exists()
    assert (payload_root / "config" / "models.yaml").exists()
    assert not (payload_root / ".venv").exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["embedded_runtime"] is False
    assert manifest["runtime_path"] is None
