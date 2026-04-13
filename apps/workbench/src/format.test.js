import test from "node:test";
import assert from "node:assert/strict";

import { humanizeRuntimeText, humanizeRuntimeToken, translateDisplayJson } from "./format.js";

test("humanizeRuntimeToken translates common backend codes into Chinese labels", () => {
  assert.equal(humanizeRuntimeToken("endogenous_replay"), "内源回放");
  assert.equal(humanizeRuntimeToken("EmergentActionSketch"), "涌现行动草图");
  assert.equal(humanizeRuntimeToken("check_relation"), "维持关系脉冲");
  assert.equal(humanizeRuntimeToken("current_pointer_recent_user_turn"), "当前指向最近用户轮次");
  assert.equal(humanizeRuntimeToken("ForcedModeSwitch"), "强制模式切换");
  assert.equal(humanizeRuntimeToken("general"), "常规语境");
  assert.equal(humanizeRuntimeToken("fallback"), "回退模型");
});

test("humanizeRuntimeText keeps technical identifiers intact while translating readable phrases", () => {
  assert.equal(humanizeRuntimeText("authenticity guard fallback"), "身份校验要求回退输出");
  assert.equal(humanizeRuntimeText("observer_heartbeat"), "观察器心跳");
  assert.equal(
    humanizeRuntimeText("你刚才提到“endogenous trigger silent_but_active”。 我还挂着“endogenous:silent_but_active”这条线。"),
    "你刚才提到“内源触发 静默但活跃”。 我还挂着“内源 · 静默但活跃”这条线。",
  );
  assert.equal(humanizeRuntimeText("/workbench/round/42"), "/workbench/round/42");
  assert.equal(humanizeRuntimeText("round://42"), "round://42");
});

test("translateDisplayJson rewrites visible keys and values for the front-end raw panel", () => {
  const payload = translateDisplayJson({
    roundId: 42,
    action: "respond",
    whyNot: {
      blocked_by: ["InitiativeRuntime"],
      summary: "authenticity guard fallback",
    },
    selected_session_reason: "current_pointer_recent_user_turn",
  });

  assert.deepEqual(payload, {
    轮次: 42,
    动作: "回应",
    未采纳路径: {
      被阻断模块: ["主动运行时"],
      摘要: "身份校验要求回退输出",
    },
    已选会话原因: "当前指向最近用户轮次",
  });
});
