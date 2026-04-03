# NALR Architecture Overview

NALR 采用 Python-first 的模块化运行时，并将 observer 保持为只读 sidecar。核心设计意图是让 proposal、状态存储、trace 和 CLI 共享同一套本地运行态目录，而不是过早拆成多服务写路径。

当前代码边界：

- `RuntimeController` 负责编排、状态持久化、命令应用和 checkpoint
- Agent 模块只返回 proposal，不直接执行动作
- `TraceStore` 只负责 round/command evidence
- `MemoryStore` 统一承接记忆、习惯、关系三个慢变量
- observer 通过读取相同的状态与 trace 文件提供诊断视图

