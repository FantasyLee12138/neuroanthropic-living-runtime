# Humanized Fallback Rendering Design

## Goal

Make user-visible fallback expression a last-resort bottom line instead of the default degraded output. When the main renderer path fails or authenticity guard rejects the renderer output, the runtime should first try a lighter model-backed humanized rendering path, and only fall back to deterministic template text if every model path fails.

## Scope

- Keep internal typed fallbacks unchanged.
- Change only user-visible rendering fallback behavior.
- Use `deepseek` and `doubao-lite` as the fallback rendering model chain.

## Design

### What stays the same

- Skill-level typed fallback for `state_patch`, `score_map`, `contribution`, `gist`, and similar contracts remains deterministic and schema-safe.
- The existing `fallback_render_text()` function remains the final deterministic bottom line.

### What changes

- Add a renderer fallback model chain for user-visible text generation.
- When `render_expression` falls back, call the humanized fallback chain before calling `fallback_render_text()`.
- When authenticity guard decides `guard_action == "fallback"`, call the same humanized fallback chain before dropping to deterministic fallback text.

### Fallback model chain

1. Try a fast fallback render route backed by `deepseek`.
2. If that fails, try a secondary small fallback render route backed by `doubao-lite`.
3. If both fail or produce unusable text, use `fallback_render_text()`.

### Prompting constraints

- Speak only from the runtime self, never as provider/model.
- Preserve existing identity and authenticity constraints.
- Keep output natural, humanized, concise, and context-aware.
- Use the deterministic fallback sentence as grounding material, not as mandatory surface text.

## Testing

- Main renderer failure should use model-backed fallback text instead of deterministic fallback text.
- Authenticity fallback should also use model-backed fallback text.
- If both fallback models fail, the runtime should still return deterministic fallback text.
