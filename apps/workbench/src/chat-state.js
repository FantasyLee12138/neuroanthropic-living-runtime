function firstText(...values) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return "";
}

function normalizeTimestamp(value) {
  const text = firstText(value);
  if (!text) {
    return "";
  }
  const parsed = Date.parse(text);
  return Number.isNaN(parsed) ? "" : new Date(parsed).toISOString();
}

function compareIsoTimestamp(left, right) {
  const leftTime = normalizeTimestamp(left);
  const rightTime = normalizeTimestamp(right);
  if (!leftTime && !rightTime) {
    return 0;
  }
  if (!leftTime) {
    return -1;
  }
  if (!rightTime) {
    return 1;
  }
  if (leftTime === rightTime) {
    return 0;
  }
  return leftTime > rightTime ? 1 : -1;
}

function messageSignature(item) {
  return `${item?.role || ""}::${firstText(item?.text)}`;
}

function messagesTailMatches(left, right) {
  const normalizedLeft = sanitizeChatMessages(left);
  const normalizedRight = sanitizeChatMessages(right);
  if (!normalizedLeft.length || !normalizedRight.length || normalizedLeft.length < normalizedRight.length) {
    return false;
  }
  const offset = normalizedLeft.length - normalizedRight.length;
  return normalizedRight.every((item, index) => messageSignature(item) === messageSignature(normalizedLeft[offset + index]));
}

function transcriptLineTimestamp(item) {
  return normalizeTimestamp(item?.timestamp || item?.ts || item?.created_at || item?.recorded_at);
}

function placeholderSessionState(chatSessionId) {
  const sessionId = firstText(chatSessionId);
  if (!sessionId) {
    return null;
  }
  return {
    session: {
      session_id: sessionId,
      transcript_lines: [],
    },
  };
}

export function transcriptToChatMessages(sessionState, previousMessages = []) {
  const previous = sanitizeChatMessages(previousMessages, { preservePending: true });
  let previousCursor = 0;
  return (sessionState?.session?.transcript_lines || [])
    .filter((item) => item && ["user", "assistant"].includes(item.kind) && firstText(item.text))
    .map((item) => {
      const role = item.kind;
      const text = firstText(item.text);
      let timestamp = transcriptLineTimestamp(item);
      if (!timestamp) {
        const nextIndex = previous.findIndex(
          (candidate, index) => index >= previousCursor && messageSignature(candidate) === `${role}::${text}`,
        );
        if (nextIndex >= 0) {
          timestamp = normalizeTimestamp(previous[nextIndex].timestamp);
          previousCursor = nextIndex + 1;
        }
      }
      return {
        role,
        text,
        timestamp,
      };
    });
}

export function sanitizeChatMessages(items, { preservePending = false } = {}) {
  return (Array.isArray(items) ? items : [])
    .filter((item) => item && ["user", "assistant"].includes(item.role) && firstText(item.text))
    .slice(-40)
    .map((item) => {
      const normalized = {
        role: item.role,
        text: firstText(item.text),
        timestamp: normalizeTimestamp(item.timestamp || item.ts || item.created_at || item.recorded_at),
        reasonSummary: firstText(item.reasonSummary),
        memoryHint: firstText(item.memoryHint),
        refs: Array.isArray(item.refs) ? item.refs.filter((ref) => firstText(ref?.label) && firstText(ref?.href)) : [],
      };
      if (preservePending && item.pending) {
        normalized.pending = true;
      }
      return normalized;
    });
}

export function resolveChatHydration({
  chatSessionId = "",
  chatSessionState = null,
  observerSessionState = null,
  inMemoryMessages = [],
  persistedMessages = [],
  localMutationAt = "",
} = {}) {
  void observerSessionState;
  const current = sanitizeChatMessages(inMemoryMessages, { preservePending: true });
  const persisted = sanitizeChatMessages(persistedMessages);
  const transcript = transcriptToChatMessages(chatSessionState, current.length ? current : persisted);
  const sessionState =
    chatSessionState ||
    (current.length || persisted.length ? placeholderSessionState(chatSessionId) : null);
  const hasPending = current.some((item) => item.pending);
  const serverUpdatedAt = normalizeTimestamp(
    chatSessionState?.session?.updated_at || chatSessionState?.session?.created_at || "",
  );
  const currentCaughtUp = current.length > 0 && messagesTailMatches(transcript, current);
  const canUseServerTranscript =
    transcript.length > 0 &&
    !hasPending &&
    (
      currentCaughtUp ||
      (!current.length && (!localMutationAt || compareIsoTimestamp(serverUpdatedAt, localMutationAt) >= 0))
    );

  if (canUseServerTranscript) {
    return {
      source: "server",
      messages: transcript,
      sessionState,
    };
  }
  if (current.length) {
    return {
      source: "memory",
      messages: current,
      sessionState,
    };
  }
  if (persisted.length) {
    return {
      source: "persisted",
      messages: persisted,
      sessionState,
    };
  }
  return {
    source: "empty",
    messages: [],
    sessionState,
  };
}

export function deriveRuntimeChatName({ subject = null, runtimeState = null, settings = null, fallbackLabel = "运行体" } = {}) {
  return firstText(
    subject?.subject_kernel?.display_name,
    runtimeState?.cognitive_snapshot?.identity?.display_name,
    settings?.identity?.display_name,
    fallbackLabel,
  );
}

export function formatChatTimestamp(timestamp, locale = "zh-CN") {
  const normalized = normalizeTimestamp(timestamp);
  if (!normalized) {
    return "";
  }
  return new Intl.DateTimeFormat(locale, {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date(normalized));
}
