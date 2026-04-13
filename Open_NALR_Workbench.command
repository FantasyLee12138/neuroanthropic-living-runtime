#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  exec "$ROOT_DIR/alive-observer" foreground --help
fi

exec "$ROOT_DIR/alive-observer" foreground "$@"
