export const WORKBENCH_READ_MODEL_PATH = "/workbench/read-model";
const ROUND_DETAIL_FALLBACK_ACTION = "respond";

export async function loadJson(path) {
  const response = await fetch(path, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`${path} -> ${response.status}`);
  }
  return response.json();
}

export async function loadText(path, accept = "text/plain") {
  const response = await fetch(path, {
    headers: { Accept: accept },
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`${path} -> ${response.status}`);
  }
  return response.text();
}

export async function postJson(path, payload = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    cache: "no-store",
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`${path} -> ${response.status}`);
  }
  return response.json();
}

export async function enqueueWorkbenchUserTurn(sessionId, text, poster = postJson) {
  return poster("/web/session/event", {
    type: "user_turn",
    session_id: String(sessionId || ""),
    text: String(text || ""),
  });
}

export function buildWorkbenchSessionEventsPath(sessionId, afterId = 0) {
  const params = new URLSearchParams({
    session_id: String(sessionId || ""),
    after_id: String(afterId || 0),
    once: "true",
  });
  return `/web/session/events?${params.toString()}`;
}

export function parseSseEventPayloads(sourceText = "") {
  return String(sourceText || "")
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data: "))
    .map((line) => JSON.parse(line.slice(6)))
    .filter((payload) => payload && typeof payload === "object");
}

export async function pollWorkbenchSessionEventsOnce(sessionId, afterId = 0, loader = loadText) {
  const sourceText = await loader(buildWorkbenchSessionEventsPath(sessionId, afterId), "text/event-stream");
  return parseSseEventPayloads(sourceText);
}

function firstObject(...values) {
  for (const value of values) {
    if (value && typeof value === "object") {
      return value;
    }
  }
  return {};
}

function firstDefined(...values) {
  for (const value of values) {
    if (value !== undefined && value !== null) {
      return value;
    }
  }
  return null;
}

function looksLikeTrace(value) {
  return Boolean(
    value &&
      typeof value === "object" &&
      (
        value.sampled_action ||
        value.rendered_expression ||
        value.probability_field ||
        value.action_bookkeeping ||
        value.gate_decisions
      ),
  );
}

export function buildWorkbenchRoundPath(roundId) {
  return `/workbench/round/${encodeURIComponent(roundId)}`;
}

export function buildWorkbenchReadModelRequestPath(options = {}) {
  const params = new URLSearchParams();
  if (options.analysis) {
    params.set("analysis", "true");
  }
  if (options.innerSpace) {
    params.set("inner_space", "true");
  }
  if (options.settings) {
    params.set("settings", "true");
  }
  if (options.observerSessionId) {
    params.set("observer_session_id", String(options.observerSessionId));
  }
  if (options.chatSessionId) {
    params.set("chat_session_id", String(options.chatSessionId));
  }
  const query = params.toString();
  return query ? `${WORKBENCH_READ_MODEL_PATH}?${query}` : WORKBENCH_READ_MODEL_PATH;
}

function hasConsolePayload(payload = {}) {
  return Boolean(
    payload &&
      typeof payload === "object" &&
      (
        Array.isArray(payload.recent_rounds) ||
        payload.state ||
        payload.action_field ||
        payload.source_links ||
        payload.why_not
      ),
  );
}

function hasUsableObject(value) {
  if (!value || typeof value !== "object") {
    return false;
  }
  if (Array.isArray(value)) {
    return value.length > 0;
  }
  return Object.keys(value).length > 0;
}

function coalesceRoundField(primary, fallback) {
  return hasUsableObject(primary) ? primary : fallback;
}

export function normalizeWorkbenchReadModel(payload = {}) {
  const primary = firstObject(payload.read_model, payload.console, payload.data?.read_model, payload.data?.console, payload.data, payload);
  if (hasConsolePayload(primary)) {
    return primary;
  }
  return firstObject(payload.console, payload.data?.console, payload);
}

export function normalizeWorkbenchRoundPayload(payload = {}) {
  const source = firstObject(payload.round, payload.data?.round, payload.data, payload);
  const probabilityField = firstObject(source.probability_field, source.trace?.probability_field);
  return {
    trace: looksLikeTrace(source.trace) ? source.trace : looksLikeTrace(source) ? source : null,
    initiativeWhy: firstDefined(source.initiativeWhy, source.initiative_why, source.initiative_why_summary),
    thought: firstDefined(source.thought, source.reasoning),
    why: firstDefined(source.why),
    contributions: firstDefined(source.contributions),
    probability: Object.keys(probabilityField).length ? { probability_field: probabilityField } : firstDefined(source.probability),
    whyNot: firstDefined(source.whyNot, source.why_not),
    replay: firstDefined(source.replay),
    raw: source,
  };
}

export async function loadWorkbenchReadModelEnvelope(optionsOrLoader = {}, maybeLoader = loadJson) {
  const options = typeof optionsOrLoader === "function" ? {} : (optionsOrLoader || {});
  const loader = typeof optionsOrLoader === "function" ? optionsOrLoader : maybeLoader;
  return loader(buildWorkbenchReadModelRequestPath(options));
}

export async function loadWorkbenchReadModelData(loader = loadJson) {
  const primary = normalizeWorkbenchReadModel(await loader(WORKBENCH_READ_MODEL_PATH));
  if (!hasConsolePayload(primary)) {
    throw new Error(`${WORKBENCH_READ_MODEL_PATH} returned an unusable payload`);
  }
  return primary;
}

export async function loadWorkbenchRoundCatalogData(roundId, loader = loadJson) {
  return normalizeWorkbenchRoundPayload(await loader(buildWorkbenchRoundPath(roundId)));
}

export async function loadWorkbenchRoundData(roundId, actionOrLoader = ROUND_DETAIL_FALLBACK_ACTION, maybeLoader = loadJson) {
  const action = typeof actionOrLoader === "function" ? ROUND_DETAIL_FALLBACK_ACTION : actionOrLoader || ROUND_DETAIL_FALLBACK_ACTION;
  const loader = typeof actionOrLoader === "function" ? actionOrLoader : maybeLoader;
  const normalized = normalizeWorkbenchRoundPayload(await loader(buildWorkbenchRoundPath(roundId)));
  const missing = [
    ["trace", normalized?.trace],
    ["initiativeWhy", normalized?.initiativeWhy],
    ["why", normalized?.why],
    ["contributions", normalized?.contributions],
    ["probability", normalized?.probability],
    ["whyNot", normalized?.whyNot],
    ["replay", normalized?.replay],
    ["thought", normalized?.thought],
  ].filter(([, value]) => !value);
  if (missing.length) {
    const missingFields = missing.map(([field]) => field).join(", ");
    throw new Error(`${buildWorkbenchRoundPath(roundId)} missing required fields: ${missingFields} (action=${action})`);
  }
  return normalized;
}

export const loadWorkbenchReadModel = loadWorkbenchReadModelData;
export const loadWorkbenchRound = loadWorkbenchRoundData;
