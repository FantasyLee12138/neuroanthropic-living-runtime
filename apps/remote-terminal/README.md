# Remote Terminal

`apps/remote-terminal/` 是一个 stateless relay client。

它不持久化运行时真相，不维护第二份 session/runtime 状态，只把当前页面渲染需要的瞬时数据保存在内存里，并把所有读写都转发到 observer 的既有 web session API：

- `POST /web/session/start`
- `POST /web/session/event`
- `GET /web/session/events`
- `GET /web/session/state`

## 使用

1. 先启动 observer 服务。
2. 直接打开 [`index.html`](/Users/fantasylee/.config/superpowers/worktrees/类脑架构/codex-cognitive-chain-control/apps/remote-terminal/index.html)。
3. 填写 observer base URL 和 session id。
4. 点击 `Start / Resume` 建立或恢复会话。
5. 用 `User Turn` 或 `Control Command` 发送输入，右侧会实时显示 transcript、events 和 session state。

## 底线

- 这是 observer 的轻量接入端，不是第二运行时。
- 不使用 `localStorage` / `sessionStorage`。
- 所有状态读写必须经过服务端唯一实例。
