from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not (REPO_ROOT / "apps" / "terminal" / "node_modules").exists(), reason="terminal dependencies not installed")
def test_repo_launcher_exposes_nalr_help():
    result = subprocess.run(
        [str(REPO_ROOT / "NALR"), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Start interactive terminal shell" in result.stdout
    assert "Slash commands:" in result.stdout


@pytest.mark.skipif(not (REPO_ROOT / "apps" / "terminal" / "node_modules").exists(), reason="terminal dependencies not installed")
def test_nalr_one_shot_runs_read_only_planning_session(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "layout.py").write_text("VALUE = 1\n", encoding="utf-8")
    env = os.environ.copy()
    env["NALR_HOME"] = str(tmp_path / ".alive")
    env["NALR_CONFIG_DIR"] = str(REPO_ROOT / "config")
    env["NALR_SKIP_ENV_AUTOLOAD"] = "1"

    result = subprocess.run(
        [str(REPO_ROOT / "NALR"), "总结这个仓库结构"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0
    assert "NALR:" in result.stdout
    assert result.stdout.count("NALR:") == 1
    assert "Run " not in result.stdout
    assert "Step:" not in result.stdout
    assert "Tool " not in result.stdout
    assert "repo_scan" not in result.stdout
    assert "已进入只读任务处理。可用 /status /why /steps /tools 查看进度。" in result.stdout


@pytest.mark.skipif(not (REPO_ROOT / "apps" / "terminal" / "node_modules").exists(), reason="terminal dependencies not installed")
def test_nalr_one_shot_greeting_returns_natural_reply_only(tmp_path):
    env = os.environ.copy()
    env["NALR_HOME"] = str(tmp_path / ".alive")
    env["NALR_CONFIG_DIR"] = str(REPO_ROOT / "config")
    env["NALR_SKIP_ENV_AUTOLOAD"] = "1"

    result = subprocess.run(
        [str(REPO_ROOT / "NALR"), "你好"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0
    assert "NALR:" in result.stdout
    assert "Run " not in result.stdout
    assert "Step:" not in result.stdout
    assert "Tool " not in result.stdout


@pytest.mark.skipif(not (REPO_ROOT / "apps" / "terminal" / "node_modules").exists(), reason="terminal dependencies not installed")
def test_nalr_one_shot_identity_compound_returns_natural_reply_only(tmp_path):
    env = os.environ.copy()
    env["NALR_HOME"] = str(tmp_path / ".alive")
    env["NALR_CONFIG_DIR"] = str(REPO_ROOT / "config")
    env["NALR_SKIP_ENV_AUTOLOAD"] = "1"

    result = subprocess.run(
        [str(REPO_ROOT / "NALR"), "你好，你是谁？你有名字吗？"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0
    assert "NALR:" in result.stdout
    assert "Run " not in result.stdout
    assert "Step:" not in result.stdout
    assert "Tool " not in result.stdout
    assert "runtime_instance" not in result.stdout


@pytest.mark.skipif(not (REPO_ROOT / "apps" / "terminal" / "node_modules").exists(), reason="terminal dependencies not installed")
def test_nalr_one_shot_task_shows_paused_confirmation_when_dirty_worktree_detected(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "nalr@example.com"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "NALR Test"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "src").mkdir()
    tracked = tmp_path / "src" / "layout.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "src/layout.py"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    tracked.write_text("VALUE = 2\n", encoding="utf-8")

    env = os.environ.copy()
    env["NALR_HOME"] = str(tmp_path / ".alive")
    env["NALR_CONFIG_DIR"] = str(REPO_ROOT / "config")
    env["NALR_SKIP_ENV_AUTOLOAD"] = "1"

    result = subprocess.run(
        [str(REPO_ROOT / "NALR"), "总结这个仓库结构"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0
    assert "任务已建立，但当前处于暂停状态。可用 /status /why 查看原因。" in result.stdout


@pytest.mark.skipif(not (REPO_ROOT / "apps" / "terminal" / "node_modules").exists(), reason="terminal dependencies not installed")
def test_nalr_one_shot_dream_command_runs_manual_dream(tmp_path):
    env = os.environ.copy()
    env["NALR_HOME"] = str(tmp_path / ".alive")
    env["NALR_CONFIG_DIR"] = str(REPO_ROOT / "config")
    env["NALR_SKIP_ENV_AUTOLOAD"] = "1"

    result = subprocess.run(
        [str(REPO_ROOT / "NALR"), "Dream", "tea"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0
    assert "已手动触发 Dream" in result.stdout
    assert "sleep_full" in result.stdout
    assert "tea" in result.stdout
