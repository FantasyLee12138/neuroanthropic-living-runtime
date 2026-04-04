# Observer Screenshot Placeholder

这里预留以下 observer 视图的截图位置：

- `/state` 与 `/trace/{round_id}`：基础状态与单轮证据
- `/diagnostics/state-delta`：展示 appraisal 输入、clip 前后 delta 与 suppression reason
- `/diagnostics/identity-blockers`：展示 `identity_score`、`naming_signal`、`continuity_signal` 与 blocker 列表
- `/diagnostics/run-contamination`：展示 `drive_source`、旧 `current_goal` 和 contamination 标记
- `/diagnostics/why-no-change/{round_ref}`：展示 `not_called / called_neutral / updated_but_clipped / updated_but_not_expressed`
- `/diagnostics/migration`：展示 `.alive` schema migration 摘要

当前 dashboard 仍是占位页，因此这份文档的截图计划优先面向 API 结果，而不是已经完成的可视化面板。
