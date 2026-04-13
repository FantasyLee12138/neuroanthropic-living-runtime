# Remote Terminal

`remote-terminal` is a stateless relay client for the observer web session API.

It keeps only transient UI state in the browser. All runtime truth comes from:

- `GET /web/session/state?session_id=...`
- `GET /web/session/events?session_id=...`
- `POST /web/session/start`
- `POST /web/session/event`

Usage:

1. Open `index.html` in a browser.
2. Enter the observer base URL, for example `http://127.0.0.1:8765`.
3. Start a new session or resume an existing session id.
4. Send `user_turn` text or `control_command` payloads and watch transcript/events update from the observer.

Notes:

- The page does not use localStorage or sessionStorage for runtime truth.
- If you open the file directly and your browser blocks cross-origin requests, serve it from the same origin as the observer or put a local proxy in front of the observer base URL.
