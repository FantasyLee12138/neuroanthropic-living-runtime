import test from "node:test";
import assert from "node:assert/strict";

import {
  deriveRuntimeChatName,
  resolveChatHydration,
  sanitizeChatMessages,
  transcriptToChatMessages,
} from "./chat-state.js";

test("transcriptToChatMessages preserves transcript timestamps and reuses earlier local timestamps when the server omits them", () => {
  const previousMessages = [
    { role: "user", text: "你好", timestamp: "2026-04-13T10:00:00.000Z" },
    { role: "assistant", text: "我在", timestamp: "2026-04-13T10:00:01.000Z" },
  ];
  const sessionState = {
    session: {
      transcript_lines: [
        { kind: "user", text: "你好" },
        { kind: "assistant", text: "我在", ts: "2026-04-13T10:00:03.000Z" },
      ],
    },
  };

  const messages = transcriptToChatMessages(sessionState, previousMessages);

  assert.deepEqual(messages, [
    { role: "user", text: "你好", timestamp: "2026-04-13T10:00:00.000Z" },
    { role: "assistant", text: "我在", timestamp: "2026-04-13T10:00:03.000Z" },
  ]);
});

test("sanitizeChatMessages keeps timestamps and optional details while trimming invalid rows", () => {
  const messages = sanitizeChatMessages([
    null,
    { role: "assistant", text: "   " },
    {
      role: "assistant",
      text: "新的回复",
      timestamp: "2026-04-13T10:05:00.000Z",
      reasonSummary: "因为当前轮次刚完成。",
      memoryHint: "挂着用户刚才的线索",
      refs: [{ label: "轮次 #12", href: "/workbench/rounds/12" }, { label: "", href: "/invalid" }],
    },
  ]);

  assert.deepEqual(messages, [
    {
      role: "assistant",
      text: "新的回复",
      timestamp: "2026-04-13T10:05:00.000Z",
      reasonSummary: "因为当前轮次刚完成。",
      memoryHint: "挂着用户刚才的线索",
      refs: [{ label: "轮次 #12", href: "/workbench/rounds/12" }],
    },
  ]);
});

test("resolveChatHydration keeps fresher in-memory messages instead of replacing them with stale transcript or persisted history", () => {
  const inMemoryMessages = [
    { role: "user", text: "你知道我是谁吗", timestamp: "2026-04-13T10:05:00.000Z" },
    { role: "assistant", text: "我记得你刚才正在追问身份线索。", timestamp: "2026-04-13T10:05:03.000Z" },
  ];

  const resolved = resolveChatHydration({
    chatSessionId: "workbench-chat",
    chatSessionState: {
      session: {
        session_id: "workbench-chat",
        updated_at: "2026-04-13T10:04:40.000Z",
        transcript_lines: [{ kind: "user", text: "旧消息" }],
      },
    },
    observerSessionState: {
      session: {
        session_id: "observer-main",
        transcript_lines: [
          { kind: "user", text: "更旧的旁路消息" },
          { kind: "assistant", text: "旧 fallback" },
        ],
      },
    },
    inMemoryMessages,
    persistedMessages: [{ role: "assistant", text: "更早的本地缓存", timestamp: "2026-04-13T10:03:00.000Z" }],
    localMutationAt: "2026-04-13T10:05:03.000Z",
  });

  assert.equal(resolved.source, "memory");
  assert.deepEqual(resolved.messages, sanitizeChatMessages(inMemoryMessages));
  assert.equal(resolved.sessionState.session.session_id, "workbench-chat");
});

test("resolveChatHydration keeps chat empty after clear when a stale pre-clear transcript arrives later", () => {
  const resolved = resolveChatHydration({
    chatSessionId: "workbench-chat",
    chatSessionState: {
      session: {
        session_id: "workbench-chat",
        updated_at: "2026-04-13T10:00:00.000Z",
        transcript_lines: [
          { kind: "user", text: "清空前的消息" },
          { kind: "assistant", text: "清空前的回复" },
        ],
      },
    },
    observerSessionState: {
      session: {
        session_id: "observer-main",
        transcript_lines: [{ kind: "assistant", text: "旧 observer transcript" }],
      },
    },
    inMemoryMessages: [],
    persistedMessages: [],
    localMutationAt: "2026-04-13T10:01:00.000Z",
  });

  assert.equal(resolved.source, "empty");
  assert.deepEqual(resolved.messages, []);
});

test("deriveRuntimeChatName prefers the runtime's own display name", () => {
  const name = deriveRuntimeChatName({
    subject: {
      subject_kernel: {
        display_name: "朝屿",
      },
    },
    runtimeState: {
      cognitive_snapshot: {
        identity: {
          display_name: "备用名字",
        },
      },
    },
    settings: {
      identity: {
        display_name: "设置名字",
      },
    },
    fallbackLabel: "运行体",
  });

  assert.equal(name, "朝屿");
});
