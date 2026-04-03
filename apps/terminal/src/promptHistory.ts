export interface PromptHistoryState {
  entries: string[];
  cursor: number;
  draft: string;
}

export function createPromptHistoryState(): PromptHistoryState {
  return {
    entries: [],
    cursor: -1,
    draft: ""
  };
}

export function commitPrompt(state: PromptHistoryState, prompt: string, limit = 100): PromptHistoryState {
  const text = prompt.trim();
  if (!text) {
    return state;
  }
  const nextEntries = [...state.entries, text];
  return {
    entries: nextEntries.slice(-Math.max(1, limit)),
    cursor: -1,
    draft: ""
  };
}

export function movePromptCursor(
  state: PromptHistoryState,
  direction: "up" | "down",
  currentInput: string
): { state: PromptHistoryState; input: string } {
  if (state.entries.length === 0) {
    return { state, input: currentInput };
  }
  if (direction === "up") {
    const nextCursor = state.cursor < 0 ? state.entries.length - 1 : Math.max(0, state.cursor - 1);
    return {
      state: {
        ...state,
        cursor: nextCursor,
        draft: state.cursor < 0 ? currentInput : state.draft
      },
      input: state.entries[nextCursor] ?? ""
    };
  }
  if (state.cursor < 0) {
    return { state, input: currentInput };
  }
  const nextCursor = state.cursor + 1;
  if (nextCursor >= state.entries.length) {
    return {
      state: {
        ...state,
        cursor: -1
      },
      input: state.draft
    };
  }
  return {
    state: {
      ...state,
      cursor: nextCursor
    },
    input: state.entries[nextCursor] ?? state.draft
  };
}
