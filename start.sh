#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PORT=8081
SERVER_SCRIPT="$SCRIPT_DIR/server.py"
PID_FILE="$SCRIPT_DIR/outputs/h3-studio-$PORT.pid"
LOG_FILE="$SCRIPT_DIR/outputs/h3-studio-$PORT.log"
URL="http://127.0.0.1:$PORT"

is_h3_server() {
    local pid="$1"
    local command

    command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    [[ "$command" == *"$SERVER_SCRIPT"* && "$command" == *"--port $PORT"* ]]
}

mkdir -p "$SCRIPT_DIR/outputs"

if [[ -f "$PID_FILE" ]]; then
    pid="$(tr -d '[:space:]' < "$PID_FILE")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && is_h3_server "$pid"; then
        echo "H3 Studio 已在运行：${URL}（PID ${pid}）"
        exit 0
    fi
    rm -f "$PID_FILE"
fi

if command -v lsof >/dev/null 2>&1; then
    listener="$(lsof -nP -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
    if [[ -n "$listener" ]]; then
        if is_h3_server "$listener"; then
            printf '%s\n' "$listener" > "$PID_FILE"
            echo "H3 Studio 已在运行：${URL}（PID ${listener}）"
            exit 0
        fi
        echo "无法启动：端口 ${PORT} 已被其他进程占用（PID ${listener}）。" >&2
        exit 1
    fi
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "无法启动：找不到 python3。" >&2
    exit 1
fi

if [[ ! -f "$SERVER_SCRIPT" || ! -f "$SCRIPT_DIR/h3-ui.html" ]]; then
    echo "无法启动：server.py 或 h3-ui.html 不存在。" >&2
    exit 1
fi

if [[ ! -x "$SCRIPT_DIR/h3" ]]; then
    echo "未找到 h3 可执行文件，正在编译……"
    make -C "$SCRIPT_DIR" h3
fi

nohup python3 -u "$SERVER_SCRIPT" --port "$PORT" \
    >> "$LOG_FILE" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$PID_FILE"

attempt=0
while (( attempt < 50 )); do
    if ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$PID_FILE"
        echo "H3 Studio 启动失败，最近日志：" >&2
        tail -n 30 "$LOG_FILE" >&2 || true
        exit 1
    fi
    if curl --silent --fail --max-time 1 "$URL/" >/dev/null 2>&1; then
        echo "H3 Studio 已启动：$URL"
        echo "日志：$LOG_FILE"
        exit 0
    fi
    sleep 0.2
    ((attempt += 1))
done

kill -TERM "$pid" 2>/dev/null || true
rm -f "$PID_FILE"
echo "H3 Studio 在 10 秒内未就绪，最近日志：" >&2
tail -n 30 "$LOG_FILE" >&2 || true
exit 1
