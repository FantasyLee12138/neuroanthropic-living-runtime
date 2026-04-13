from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import hashlib
import re

from nalr.schemas.models import RunRequest, RunState, TaskNode, ToolResult


def _safe_excerpt(value: str, *, limit: int = 240) -> str:
    collapsed = " ".join(value.split())
    return collapsed[:limit]


class SupervisorLoop:
    def __init__(
        self,
        *,
        project_root: Path,
        planner: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.planner = planner

    def _tokenize_goal(self, goal: str) -> list[str]:
        lowered = goal.lower()
        tokens = re.findall(r"[a-zA-Z0-9_./-]+|[\u4e00-\u9fff]{2,}", lowered)
        return [token for token in tokens if len(token.strip()) >= 2]

    def _iter_repo_files(self) -> list[Path]:
        ignored_roots = {".git", ".venv", ".alive", "__pycache__", ".pytest_cache"}
        files: list[Path] = []
        for path in self.project_root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in ignored_roots for part in path.parts):
                continue
            files.append(path)
        return files

    def _match_files(self, goal: str, files: list[Path]) -> list[str]:
        tokens = self._tokenize_goal(goal)
        scored: list[tuple[int, str]] = []
        for path in files:
            rel = path.relative_to(self.project_root).as_posix()
            haystack = rel.lower()
            score = sum(1 for token in tokens if token in haystack)
            if score > 0:
                scored.append((score, rel))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [rel for _, rel in scored[:8]]

    def _repo_scan(self, goal: str) -> ToolResult:
        files = self._iter_repo_files()
        matched = self._match_files(goal, files)
        summary = f"scanned {len(files)} files"
        if matched:
            summary += f"; matched {len(matched)} relevant paths"
        return ToolResult(
            tool_name="repo_scan",
            status="ok",
            summary=summary,
            output_excerpt=_safe_excerpt(" | ".join(matched) if matched else "no direct file matches"),
            metadata={"matched_files": matched, "file_count": len(files)},
        )

    def _fallback_plan(self, goal: str, matched: list[str] | None = None) -> dict[str, Any]:
        matched = list(matched or [])
        if matched:
            title = f"检查相关文件：{matched[0]}"
            detail = f"先阅读命中的文件，再决定后续是否需要搜索更多上下文。"
        else:
            title = "扫描仓库入口并缩小范围"
            detail = "先定位最相关的目录、文件和测试入口。"
        return {
            "goal_summary": goal[:80],
            "next_step": title,
            "detail": detail,
            "expected_observation": "得到与目标最相关的文件、目录或测试线索",
            "success_criteria": "至少定位到一组可继续深入的仓库证据",
            "tool_choice": "repo_scan",
            "confidence": 0.62,
        }

    def prepare_bootstrap(self, request: RunRequest, *, session_id: str, recorded_at: str) -> tuple[RunState, dict[str, Any], dict[str, Any]]:
        plan = self._fallback_plan(request.goal, matched=None)
        if self.planner is not None:
            try:
                model_plan = self.planner(request.goal, {})
            except Exception:
                model_plan = None
            if isinstance(model_plan, dict):
                filtered = {
                    key: value
                    for key, value in model_plan.items()
                    if value is not None and value != "" and value != []
                }
                plan = {**plan, **filtered}

        step = TaskNode(
            node_id=f"step-{hashlib.sha1(request.goal.encode('utf-8')).hexdigest()[:8]}",
            title=str(plan["next_step"]),
            detail=str(plan.get("detail", "")),
            tool_choice=str(plan.get("tool_choice", "repo_scan")),
            status="running",
            expected_observation=str(plan.get("expected_observation", "")),
            success_criteria=str(plan.get("success_criteria", "")),
            confidence=float(plan.get("confidence", 0.5)),
            metadata={"matched_files": []},
        )
        run_state = RunState(
            goal=request.goal,
            goal_summary=str(plan.get("goal_summary", request.goal[:80])),
            status="running",
            current_step_id=step.node_id,
            pending_steps=[step],
            completed_steps=[],
            last_tool_result=None,
            policy={
                "allow_commit": request.allow_commit,
                "operator_level": request.operator_level,
                "dirty_worktree_policy": "pause",
                "pause_on_commit_boundary": True,
                "continue_on_recoverable_failure": True,
                "max_retries_per_step": 2,
            },
            budget={"max_steps": 12, "max_retries_per_step": 2},
            dirty_worktree_detected=False,
            commit_permission_required=not request.allow_commit,
            created_at=recorded_at,
            updated_at=recorded_at,
            session_id=session_id,
        )
        step_trace = {
            "run_id": run_state.run_id,
            "step_id": step.node_id,
            "title": step.title,
            "detail": step.detail,
            "status": step.status,
            "tool_choice": step.tool_choice,
            "expected_observation": step.expected_observation,
            "success_criteria": step.success_criteria,
            "confidence": step.confidence,
            "matched_files": step.metadata.get("matched_files", []),
        }
        tool_trace = {
            "run_id": run_state.run_id,
            "tool_name": str(plan.get("tool_choice", "repo_scan")),
            "status": "awaiting_approval",
            "summary": f"{str(plan.get('tool_choice', 'repo_scan'))} requires operator approval",
            "output_excerpt": "",
            "input": {"goal": request.goal},
            "matched_files": [],
            "file_count": 0,
        }
        return run_state, step_trace, tool_trace

    def execute_prepared_tool(self, tool_name: str, *, goal: str) -> ToolResult:
        if tool_name == "repo_scan":
            return self._repo_scan(goal)
        raise ValueError(f"unsupported prepared tool: {tool_name}")

    def bootstrap(self, request: RunRequest, *, session_id: str, recorded_at: str) -> tuple[RunState, dict[str, Any], dict[str, Any]]:
        tool_result = self._repo_scan(request.goal)
        plan = self._fallback_plan(
            request.goal,
            matched=list(tool_result.metadata.get("matched_files", []) or []),
        )
        if self.planner is not None:
            try:
                model_plan = self.planner(request.goal, tool_result.metadata)
            except Exception:
                model_plan = None
            if isinstance(model_plan, dict):
                filtered = {
                    key: value
                    for key, value in model_plan.items()
                    if value is not None and value != "" and value != []
                }
                plan = {**plan, **filtered}

        step = TaskNode(
            node_id=f"step-{hashlib.sha1(request.goal.encode('utf-8')).hexdigest()[:8]}",
            title=str(plan["next_step"]),
            detail=str(plan.get("detail", "")),
            tool_choice=str(plan.get("tool_choice", "repo_scan")),
            status="running",
            expected_observation=str(plan.get("expected_observation", "")),
            success_criteria=str(plan.get("success_criteria", "")),
            confidence=float(plan.get("confidence", 0.5)),
            metadata={"matched_files": tool_result.metadata.get("matched_files", [])},
        )
        run_state = RunState(
            goal=request.goal,
            goal_summary=str(plan.get("goal_summary", request.goal[:80])),
            status="running",
            current_step_id=step.node_id,
            pending_steps=[step],
            completed_steps=[],
            last_tool_result=tool_result,
            policy={
                "allow_commit": request.allow_commit,
                "operator_level": request.operator_level,
                "dirty_worktree_policy": "pause",
                "pause_on_commit_boundary": True,
                "continue_on_recoverable_failure": True,
                "max_retries_per_step": 2,
            },
            budget={"max_steps": 12, "max_retries_per_step": 2},
            dirty_worktree_detected=False,
            commit_permission_required=not request.allow_commit,
            created_at=recorded_at,
            updated_at=recorded_at,
            session_id=session_id,
        )
        step_trace = {
            "run_id": run_state.run_id,
            "step_id": step.node_id,
            "title": step.title,
            "detail": step.detail,
            "status": step.status,
            "tool_choice": step.tool_choice,
            "expected_observation": step.expected_observation,
            "success_criteria": step.success_criteria,
            "confidence": step.confidence,
            "matched_files": step.metadata.get("matched_files", []),
        }
        tool_trace = {
            "run_id": run_state.run_id,
            "tool_name": tool_result.tool_name,
            "status": tool_result.status,
            "summary": tool_result.summary,
            "output_excerpt": tool_result.output_excerpt,
            "matched_files": tool_result.metadata.get("matched_files", []),
            "file_count": tool_result.metadata.get("file_count", 0),
        }
        return run_state, step_trace, tool_trace
