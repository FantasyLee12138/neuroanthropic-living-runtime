import test from "node:test";
import assert from "node:assert/strict";

import {
  WORKBENCH_READ_MODEL_PATH,
  buildWorkbenchReadModelRequestPath,
  buildWorkbenchSessionEventsPath,
  enqueueWorkbenchUserTurn,
  loadWorkbenchReadModelEnvelope,
  buildWorkbenchRoundPath,
  loadWorkbenchReadModelData,
  loadWorkbenchRoundCatalogData,
  loadWorkbenchRoundData,
  parseSseEventPayloads,
  pollWorkbenchSessionEventsOnce,
} from "./api.js";

test("buildWorkbenchReadModelRequestPath includes query flags for the aggregated workbench payload", () => {
  const path = buildWorkbenchReadModelRequestPath({
    analysis: true,
    innerSpace: true,
    settings: true,
    observerSessionId: "observer-main",
    chatSessionId: "workbench-chat",
  });

  assert.equal(
    path,
    "/workbench/read-model?analysis=true&inner_space=true&settings=true&observer_session_id=observer-main&chat_session_id=workbench-chat",
  );
});

test("loadWorkbenchReadModelEnvelope requests the aggregated workbench payload with query params", async () => {
  const calls = [];
  const fetchJson = async (path) => {
    calls.push(path);
    return {
      bootstrap: { session_attached: true },
      summary: { runtime_state: { runtime_revision: 3 } },
      console: { recent_rounds: [{ round_id: 3, sampled_action: "respond" }] },
    };
  };

  const payload = await loadWorkbenchReadModelEnvelope(
    {
      analysis: true,
      innerSpace: true,
      chatSessionId: "workbench-chat",
    },
    fetchJson,
  );

  assert.deepEqual(calls, ["/workbench/read-model?analysis=true&inner_space=true&chat_session_id=workbench-chat"]);
  assert.equal(payload.bootstrap.session_attached, true);
  assert.equal(payload.summary.runtime_state.runtime_revision, 3);
  assert.equal(payload.console.recent_rounds[0].round_id, 3);
});

test("loadWorkbenchReadModelData prefers the workbench read-model endpoint", async () => {
  const calls = [];
  const fetchJson = async (path) => {
    calls.push(path);
    if (path === WORKBENCH_READ_MODEL_PATH) {
      return { read_model: { recent_rounds: [{ round_id: 42, sampled_action: "respond" }], state: { current_round: { round_id: 42 } } } };
    }
    throw new Error(`unexpected ${path}`);
  };

  const payload = await loadWorkbenchReadModelData(fetchJson);

  assert.deepEqual(calls, [WORKBENCH_READ_MODEL_PATH]);
  assert.equal(payload.recent_rounds[0].round_id, 42);
  assert.equal(payload.state.current_round.round_id, 42);
});

test("loadWorkbenchReadModelData does not fall back to legacy console refresh when the aggregated endpoint fails", async () => {
  const calls = [];
  const fetchJson = async (path) => {
    calls.push(path);
    if (path === WORKBENCH_READ_MODEL_PATH) {
      throw new Error("404");
    }
    throw new Error(`unexpected ${path}`);
  };

  await assert.rejects(() => loadWorkbenchReadModelData(fetchJson), /404/);
  assert.deepEqual(calls, [WORKBENCH_READ_MODEL_PATH]);
});

test("loadWorkbenchRoundData normalizes aggregated round payloads from the new endpoint", async () => {
  const calls = [];
  const fetchJson = async (path) => {
    calls.push(path);
    if (path === buildWorkbenchRoundPath(18)) {
      return {
        round: {
          trace: { sampled_action: "respond" },
          why: { why: { summary: "新接口摘要" } },
          why_not: { summary: "未采纳路径" },
          initiative_why: { summary: "主动性判断" },
          contributions: { stacked_contributions: [] },
          probability_field: { action: { winner_posterior: { respond: 0.91 } } },
          replay: { replayed_action: "wait" },
          thought: { focus: "memory" },
        },
      };
    }
    throw new Error(`unexpected ${path}`);
  };

  const payload = await loadWorkbenchRoundData(18, fetchJson);

  assert.deepEqual(calls, [buildWorkbenchRoundPath(18)]);
  assert.equal(payload.trace.sampled_action, "respond");
  assert.equal(payload.why.why.summary, "新接口摘要");
  assert.equal(payload.whyNot.summary, "未采纳路径");
  assert.equal(payload.initiativeWhy.summary, "主动性判断");
  assert.equal(payload.probability.probability_field.action.winner_posterior.respond, 0.91);
  assert.equal(payload.replay.replayed_action, "wait");
  assert.equal(payload.thought.focus, "memory");
});

test("loadWorkbenchRoundCatalogData does not fall back to legacy round endpoints when the aggregated endpoint fails", async () => {
  const calls = [];
  const fetchJson = async (path) => {
    calls.push(path);
    if (path === buildWorkbenchRoundPath(29)) {
      throw new Error("404");
    }
    throw new Error(`unexpected ${path}`);
  };

  await assert.rejects(() => loadWorkbenchRoundCatalogData(29, fetchJson), /404/);
  assert.deepEqual(calls, [buildWorkbenchRoundPath(29)]);
});

test("loadWorkbenchRoundData does not fall back to legacy round endpoints when the aggregated endpoint fails", async () => {
  const calls = [];
  const fetchJson = async (path) => {
    calls.push(path);
    if (path === buildWorkbenchRoundPath(29)) {
      throw new Error("404");
    }
    throw new Error(`unexpected ${path}`);
  };

  await assert.rejects(() => loadWorkbenchRoundData(29, fetchJson), /404/);
  assert.deepEqual(calls, [buildWorkbenchRoundPath(29)]);
});

test("enqueueWorkbenchUserTurn posts chat input into the web session queue", async () => {
  const calls = [];
  const post = async (path, payload) => {
    calls.push({ path, payload });
    return { accepted: true, last_event_id: 9, queued: true };
  };

  const response = await enqueueWorkbenchUserTurn("workbench-chat", "你知道我是谁吗", post);

  assert.deepEqual(calls, [
    {
      path: "/web/session/event",
      payload: {
        type: "user_turn",
        session_id: "workbench-chat",
        text: "你知道我是谁吗",
      },
    },
  ]);
  assert.equal(response.accepted, true);
  assert.equal(response.queued, true);
  assert.equal(response.last_event_id, 9);
});

test("buildWorkbenchSessionEventsPath encodes session polling parameters", () => {
  const path = buildWorkbenchSessionEventsPath("workbench-chat", 17);

  assert.equal(path, "/web/session/events?session_id=workbench-chat&after_id=17&once=true");
});

test("parseSseEventPayloads keeps only JSON data lines from the event stream", () => {
  const events = parseSseEventPayloads([
    "id: 18",
    'data: {"event_id":18,"type":"assistant_token","delta":"我知道"}',
    "",
    ": keep-alive",
    'data: {"event_id":19,"type":"assistant_final","message":"我知道现在正在和我说话的是你。"}',
    "",
  ].join("\n"));

  assert.deepEqual(events, [
    { event_id: 18, type: "assistant_token", delta: "我知道" },
    { event_id: 19, type: "assistant_final", message: "我知道现在正在和我说话的是你。" },
  ]);
});

test("pollWorkbenchSessionEventsOnce requests the session event stream once and parses the batch", async () => {
  const calls = [];
  const loadText = async (path) => {
    calls.push(path);
    return [
      'data: {"event_id":22,"type":"assistant_token","delta":"我知道"}',
      "",
      'data: {"event_id":23,"type":"assistant_final","message":"我知道现在正在和我说话的是你。"}',
      "",
    ].join("\n");
  };

  const events = await pollWorkbenchSessionEventsOnce("workbench-chat", 21, loadText);

  assert.deepEqual(calls, ["/web/session/events?session_id=workbench-chat&after_id=21&once=true"]);
  assert.deepEqual(events, [
    { event_id: 22, type: "assistant_token", delta: "我知道" },
    { event_id: 23, type: "assistant_final", message: "我知道现在正在和我说话的是你。" },
  ]);
});
