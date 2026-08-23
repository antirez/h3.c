#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PORT=8081
SERVER_SCRIPT="$SCRIPT_DIR/server.py"
PID_FILE="$SCRIPT_DIR/outputs/h3-studio-$PORT.pid"

is_h3_server() {
    local pid="$1"
    local command

    command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    [[ "$command" == *"$SERVER_SCRIPT"* && "$command" == *"--port $PORT"* ]]
}

pid=""
if [[ -f "$PID_FILE" ]]; then
    candidate="$(tr -d '[:space:]' < "$PID_FILE")"
    if [[ "$candidate" =~ ^[0-9]+$ ]] && kill -0 "$candidate" 2>/dev/null && is_h3_server "$candidate"; then
        pid="$candidate"
    else
        rm -f "$PID_FILE"
    fi
fi

if [[ -z "$pid" ]] && command -v lsof >/dev/null 2>&1; then
    while IFS= read -r candidate; do
        if [[ -n "$candidate" ]] && is_h3_server "$candidate"; then
            pid="$candidate"
            break
        fi
    done < <(lsof -nP -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)
fi

if [[ -z "$pid" ]]; then
    echo "H3 Studio 未在端口 $PORT 运行。"
    exit 0
fi

kill -TERM "$pid"

attempt=0
while (( attempt < 50 )); do
    if ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$PID_FILE"
        echo "H3 Studio 已停止（PID ${pid}）。"
        exit 0
    fi
    sleep 0.2
    ((attempt += 1))
done

echo "优雅停止超时，发送 SIGKILL……" >&2
kill -KILL "$pid" 2>/dev/null || true

attempt=0
while (( attempt < 25 )); do
    if ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$PID_FILE"
        echo "H3 Studio 已停止（PID ${pid}）。"
        exit 0
    fi
    sleep 0.2
    ((attempt += 1))
done

echo "无法停止 H3 Studio（PID ${pid}），请检查该进程。" >&2
exit 1
