# Project Mindmap

下面这张图按当前仓库状态整理项目主结构，重点突出运行时主链、观测层和终端层之间的关系。

```mermaid
mindmap
  root((NALR))
    Entry Points
      "./NALR"
      "./alive"
      "observer API"
    Runtime Core
      "RuntimeController"
      "IdentityRuntime"
      "AuthenticityPolicy"
      "VitalityEngine"
      "LongRunAnalyzer"
    Agents
      "PFCAgent"
      "ConflictMonitorAgent"
      "ThalamusAttentionAgent"
      "BodyStateAgent"
      "EmotionAgent"
      "RelationshipAgent"
      "ResourceAgent"
      "UnconsciousAgent"
      "DMN / Habit / Desire / Hippocampus"
    Closed Loop
      "input / event"
      "appraisal"
      "state update"
      "memory / relation writeback"
      "chronic signal"
      "identity evidence"
      "expression / renderer"
      "trace / diagnostics"
    Memory And Slow Variables
      "MemoryStore"
      "stable priors"
      "habits"
      "relation"
      "hot / warm / archive"
      "cue normalization"
      "migration"
      "temperament drift"
    Trace And Storage
      "RoundTrace"
      "TraceStore"
      "JSON / JSONL audit"
      "Parquet live read"
      "command trace"
      "repair trace"
    Observer
      "/state"
      "/trace/{round_id}"
      "/why/{round_id}"
      "/metrics/*"
      "/diagnostics/*"
      "/dashboard (placeholder)"
    Terminal Layer
      "terminal_bridge"
      "streamed assistant_token"
      "session persistence"
      "tool timeline"
      "approvals"
      "NALR control console"
    Output Layer
      "style profile"
      "render plan"
      "renderer"
      "identity explanation"
      "fallback text"
    Dreams And Noninteractive
      "DreamOrchestrator"
      "idle / sleep shaping"
      "dream trace"
      "memory consolidation"
    Interfaces
      "CLI / CIL"
      "typed command envelope"
      "checkpoint / rewind"
      "memory / trace maintenance"
    Providers And Skills
      "SkillExecutor"
      "typed contracts"
      "policy_check"
      "fallback routing"
      "circuit breaker"
      "ModelRouter"
    Tests And Validation
      "unit"
      "integration"
      "simulation"
      "longrun"
    Docs
      "README"
      "architecture overview"
      "active specs"
      "historical designs"
```

如果后面要继续扩展，这张图最容易再拆成三张专题图：

- 运行时闭环图
- observer / diagnostics 图
- 终端 / bridge / session 图
