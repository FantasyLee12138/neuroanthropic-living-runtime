# ADR 0001: Python-first Runtime

## Context

v0.56 首版要求本地可运行、可追踪、可调参与 CLI 友好。当前更重要的是数值规则、文件化状态和快速试验，而不是先做分布式拆分。

## Decision

核心 runtime 使用 Python 实现，observer 保持只读 sidecar。Web dashboard 可以后续扩展，但不反向控制 runtime。

## Consequences

- 便于快速实现规则/状态/trace
- CLI、测试和 observer 可以共享一套类型与目录
- 后续若要拆服务，应保留当前文件接口作为最小兼容层

