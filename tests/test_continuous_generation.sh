#!/usr/bin/env bash
#
# test_continuous_generation.sh — generate_series.sh / concat_segments.sh 的测试。
# 需要 PATH 中的 ffmpeg / ffprobe（本仓库生成链路本来就依赖它们），以及 python3。
# 用法：bash tests/test_continuous_generation.sh
#
# 覆盖：语法、无参数用法、mock h3 下的生成/中断/重试/续跑/透传参数、
#       concat 无损拼接 / 重编码 / 损坏跳过。
#
# 说明：generate_series.sh 是无限循环，测试通过 tests/sigint_helper.py 模拟
#       Ctrl+C（真实后台 shell 会忽略 SIGINT，故用 python fork 启动脚本）。

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GS="$ROOT/generate_series.sh"
CS="$ROOT/concat_segments.sh"
HELPER="$ROOT/tests/sigint_helper.py"

command -v python3 >/dev/null 2>&1 || { echo "需要 python3" >&2; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { echo "需要 ffmpeg" >&2; exit 1; }
command -v ffprobe >/dev/null 2>&1 || { echo "需要 ffprobe" >&2; exit 1; }

TMP="${TMPDIR:-/tmp}/h3_script_tests_$$"
rm -rf "$TMP"
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT

PASS=0
FAIL=0
ok()  { PASS=$((PASS+1)); printf 'ok   - %s\n' "$1"; }
bad() { FAIL=$((FAIL+1)); printf 'FAIL - %s\n' "$1"; }

check() { # check <desc> <cmd...>
    local desc="$1"; shift
    if "$@" >/dev/null 2>&1; then ok "$desc"; else bad "$desc"; fi
}

# 生成一个真实的极小 mp4 片段（16x16，0.1s，与 mock h3 使用相同参数）
mkseg() {
    ffmpeg -y -v error -f lavfi -i color=c=black:s=16x16:d=0.1 \
        -c:v libx264 -pix_fmt yuv420p -an "$1"
}

dur() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1" 2>/dev/null || echo 0; }

# approx <actual> <expected> <tolerance>：浮点容差比较
approx() {
    awk -v a="$1" -v e="$2" -v t="$3" \
        'BEGIN { d = a - e; if (d < 0) d = -d; exit !(d <= t) }'
}

# 生成 mock h3：从命令行提取 -o 输出路径，写一个真实极小 mp4；
# 通过环境变量模拟失败 / 延迟 / 损坏 / 记录参数。
make_mock_h3() {
    local dir="$1"
    mkdir -p "$dir/MiniMax-H3"
    cat > "$dir/h3" <<'MOCKEOF'
#!/usr/bin/env bash
if [ -n "${MOCK_LOG:-}" ]; then
    printf '%s\n' "$*" >> "$MOCK_LOG"
fi
out=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        -o|--output) out="$2"; shift 2 ;;
        -d|--model-dir|-p|--prompt|--first-frame|--ref-silent-video) shift 2 ;;
        --width=*|--height=*|--frames=*|--steps=*|--layers=*|--reuse=*) shift ;;
        *) shift ;;
    esac
done
[ -n "$out" ] || { echo "mock: missing -o" >&2; exit 2; }
base="$(basename "$out")"; num="${base#segment_}"; num="${num%.mp4}"

delay=0
if [ -n "${MOCK_DELAY_FROM:-}" ]; then
    [ "${MOCK_DELAY_FROM:-999999}" -le "$num" ] && delay="${MOCK_DELAY:-30}"
elif [ "${MOCK_DELAY:-0}" -gt 0 ]; then
    delay="$MOCK_DELAY"
fi

# 指定编号起先写损坏文件再长睡，模拟"写到一半被中断"
if [ -n "${MOCK_CORRUPT_FROM:-}" ] && [ "${MOCK_CORRUPT_FROM:-999999}" -le "$num" ]; then
    printf 'not-a-real-mp4-garbage' > "$out"
    sleep "${MOCK_CORRUPT_DELAY:-${MOCK_DELAY:-30}}"
    echo "mock: wrote corrupt file" >&2
    exit 0
fi

# 指定编号起先写完整有效文件再长睡，模拟"h3 已写完但进程未退出时被中断"
if [ -n "${MOCK_HOLD_FROM:-}" ] && [ "${MOCK_HOLD_FROM:-999999}" -le "$num" ]; then
    ffmpeg -y -v error -f lavfi -i color=c=black:s=16x16:d=0.1 \
        -c:v libx264 -pix_fmt yuv420p -an "$out"
    sleep "${MOCK_HOLD_DELAY:-30}"
    echo "mock: held after writing valid file" >&2
    exit 0
fi

if [ "${MOCK_FAIL_ALWAYS:-0}" = "1" ]; then
    echo "mock: forced failure" >&2; exit 1
fi
if [ "${MOCK_FAIL_ONCE:-0}" = "1" ] && [ ! -e "${MOCK_FAIL_MARKER:-}" ]; then
    touch "${MOCK_FAIL_MARKER:-}"
    echo "mock: simulated first-attempt failure" >&2; exit 1
fi

[ "$delay" -gt 0 ] && sleep "$delay"
ffmpeg -y -v error -f lavfi -i color=c=black:s=16x16:d=0.1 \
    -c:v libx264 -pix_fmt yuv420p -an "$out"
MOCKEOF
    chmod +x "$dir/h3"
}

# 等待文件出现，超时 N 秒
wait_file() { # wait_file <path> <seconds>
    local path="$1" secs="$2" i
    for ((i = 0; i < secs * 5; i++)); do
        [ -f "$path" ] && return 0
        sleep 0.2
    done
    return 1
}

# 等待文件包含某模式，超时 N 秒
wait_grep() { # wait_grep <file> <pattern> <seconds>
    local file="$1" pat="$2" secs="$3" i
    for ((i = 0; i < secs * 10; i++)); do
        grep -q "$pat" "$file" 2>/dev/null && return 0
        sleep 0.1
    done
    return 1
}

# 通过 sigint_helper.py 后台启动 generate_series.sh（模拟 Ctrl+C 需后台 + 触发文件）。
# 用法：run_gen <wd> <prefix> <timeout> -- <env...> -- <script args...>
# 之后：touch $TMP/<prefix>.trigger 发送 Ctrl+C；wait_gen <prefix> <secs> 等待结束。
run_gen() {
    local wd="$1" prefix="$2" timeout="$3"; shift 3
    local envargs=()
    local scriptargs=()
    local mode=env
    for a in "$@"; do
        if [ "$a" = "--" ]; then mode=script; continue; fi
        if [ "$mode" = env ]; then envargs+=("$a"); else scriptargs+=("$a"); fi
    done
    python3 "$HELPER" --cwd "$wd" --log "$wd/gen.log" \
        --trigger "$TMP/$prefix.trigger" --exitfile "$TMP/$prefix.exit" \
        --pidfile "$TMP/$prefix.pid" --timeout "$timeout" \
        "${envargs[@]}" -- ./generate_series.sh "${scriptargs[@]}" &
    echo "$!" > "$TMP/$prefix.hpid"
}

wait_gen() { # wait_gen <prefix> <seconds>
    local prefix="$1" secs="$2" i
    local hpid
    hpid="$(cat "$TMP/$prefix.hpid" 2>/dev/null)"
    for ((i = 0; i < secs * 10; i++)); do
        if [ -f "$TMP/$prefix.exit" ]; then
            [ -n "$hpid" ] && wait "$hpid" 2>/dev/null || true
            return 0
        fi
        if [ -n "$hpid" ] && ! kill -0 "$hpid" 2>/dev/null; then
            return 1
        fi
        sleep 0.1
    done
    # 超时：终止 helper（若仍在），避免无限等待
    if [ -n "$hpid" ] && kill -0 "$hpid" 2>/dev/null; then
        kill "$hpid" 2>/dev/null || true
        wait "$hpid" 2>/dev/null || true
    fi
    return 1
}

gen_exit() { cat "$TMP/$1.exit" 2>/dev/null; }

printf '%s\n' "== 1. 语法与基础校验 =="

check "bash -n generate_series.sh" bash -n "$GS"
check "bash -n concat_segments.sh" bash -n "$CS"
check "bash -n tests/test_continuous_generation.sh" bash -n "$0"
check "生成脚本可执行" test -x "$GS"
check "拼接脚本可执行" test -x "$CS"

out="$(bash "$GS" 2>&1)" || true
printf '%s' "$out" | grep -q '用法' \
    && ok "无参数时打印用法" || bad "无参数时打印用法"
bash "$GS" >/dev/null 2>&1
[ "$?" -ne 0 ] && ok "无参数时以非零退出" || bad "无参数时以非零退出"

printf '%s\n' "== 2. generate_series.sh：连续生成 + 中断 =="

# 首两段快速生成，第 3 段先写损坏文件再长睡，便于验证中断后清理该不完整段
WD="$TMP/basic"; mkdir -p "$WD/out"; printf 'fake-png' > "$WD/start.png"
make_mock_h3 "$WD"; cp "$GS" "$WD/"
LOG="$WD/h3.log"
run_gen "$WD" basic 40 \
    MOCK_CORRUPT_FROM=3 MOCK_CORRUPT_DELAY=30 MOCK_LOG="$LOG" \
    -- -i start.png -d out -p "PROMPT" -c "CONT" -w 512 -h 512 \
    --steps 2 --frames 5 --seed 7
wait_file "$WD/out/segment_0002.mp4" 20 || bad "segment_0002 未生成"
sleep 1
: > "$TMP/basic.trigger"
wait_gen basic 25

[ "$(gen_exit basic)" = "130" ] \
    && ok "中断后退出码 130（rc=$(gen_exit basic)）" || bad "中断后退出码 130（rc=$(gen_exit basic)）"
ffprobe -v error "$WD/out/segment_0001.mp4" >/dev/null 2>&1 \
    && ok "segment_0001 完整保留" || bad "segment_0001 完整保留"
ffprobe -v error "$WD/out/segment_0002.mp4" >/dev/null 2>&1 \
    && ok "segment_0002 完整保留" || bad "segment_0002 完整保留"
[ -f "$WD/out/segment_0003.mp4" ] \
    && bad "被中断的 segment_0003（已写入损坏文件）应被删除" \
    || ok "被中断的 segment_0003（已写入损坏文件）已删除"
grep -q '中断收尾：删除不完整片段' "$WD/gen.log" \
    && ok "中断收尾删除写入一半的片段" || bad "中断收尾删除写入一半的片段"

grep -q '^\[1\] prompt=PROMPT' "$WD/gen.log" \
    && ok "首段日志显示首段提示词" || bad "首段日志显示首段提示词"
grep -q '^\[2\] prompt=CONT' "$WD/gen.log" \
    && ok "第 2 段日志显示延续提示词 CONT" || bad "第 2 段日志显示延续提示词"
grep -q '预计约 30.5 分钟' "$WD/gen.log" \
    && ok "首段打印预计耗时（30.5 分钟）" || bad "首段打印预计耗时"
grep -q '预计约 32 分钟' "$WD/gen.log" \
    && ok "后续段打印预计耗时（32 分钟）" || bad "后续段打印预计耗时"
grep -q -- '--first-frame start.png' "$LOG" \
    && ok "首段使用 --first-frame 锚定起始图" || bad "首段使用 --first-frame"
grep -q -- '--ref-silent-video out/segment_0001.mp4' "$LOG" \
    && ok "第 2 段使用 --ref-silent-video 参考上一段" || bad "第 2 段使用 --ref-silent-video"
grep -q -- '--seed 7' "$LOG" \
    && ok "EXTRA_ARGS 透传 --seed 7 生效" || bad "EXTRA_ARGS 透传 --seed 7"
grep -q 'concat_segments.sh' "$WD/gen.log" \
    && ok "中断收尾打印拼接命令提示" || bad "中断收尾打印拼接命令提示"
grep -q '已中断。当前保留的有效片段' "$WD/gen.log" \
    && ok "中断收尾列出保留片段" || bad "中断收尾列出保留片段"

printf '%s\n' "== 3. generate_series.sh：中断时清理不完整片段 =="

# 第 2 段 mock 先写垃圾文件再长睡 → 中断时脚本应删除该不完整片段
WD="$TMP/interrupt"; mkdir -p "$WD/out"; printf 'fake-png' > "$WD/start.png"
make_mock_h3 "$WD"; cp "$GS" "$WD/"
LOG="$WD/h3.log"
run_gen "$WD" interrupt 40 \
    MOCK_CORRUPT_FROM=2 MOCK_CORRUPT_DELAY=30 MOCK_LOG="$LOG" \
    -- -i start.png -d out -p "PROMPT" -c "CONT" --steps 2 --frames 5
wait_file "$WD/out/segment_0001.mp4" 20 || bad "interrupt: segment_0001 未生成"
sleep 1.5   # 让第 2 段 mock 写入垃圾文件并进入长睡
: > "$TMP/interrupt.trigger"
wait_gen interrupt 25

[ "$(gen_exit interrupt)" = "130" ] \
    && ok "中断清理场景退出码 130" || bad "中断清理场景退出码（rc=$(gen_exit interrupt)）"
ffprobe -v error "$WD/out/segment_0001.mp4" >/dev/null 2>&1 \
    && ok "中断清理后 segment_0001 保留" || bad "中断清理后 segment_0001 保留"
[ -f "$WD/out/segment_0002.mp4" ] \
    && bad "不完整的 segment_0002 应被删除" || ok "不完整的 segment_0002 已删除"
grep -q '中断收尾：删除不完整片段' "$WD/gen.log" \
    && ok "打印删除不完整片段提示" || bad "打印删除不完整片段提示"

printf '%s\n' "== 4. generate_series.sh：校验窗口内不误删（完整片段保留）=="

# 第 2 段 mock 写完完整有效文件后进程未退出（模拟校验窗口），
# 此时中断 → 收尾 ffprobe 校验通过，应保留该完整片段
WD="$TMP/hold"; mkdir -p "$WD/out"; printf 'fake-png' > "$WD/start.png"
make_mock_h3 "$WD"; cp "$GS" "$WD/"
LOG="$WD/h3.log"
run_gen "$WD" hold 40 \
    MOCK_HOLD_FROM=2 MOCK_HOLD_DELAY=30 MOCK_LOG="$LOG" \
    -- -i start.png -d out -p "PROMPT" -c "CONT" --steps 2 --frames 5
wait_file "$WD/out/segment_0001.mp4" 20 || bad "hold: segment_0001 未生成"
sleep 1.5   # 第 2 段 mock 写完有效文件并进入长睡
: > "$TMP/hold.trigger"
wait_gen hold 25

[ "$(gen_exit hold)" = "130" ] \
    && ok "校验窗口中断退出码 130" || bad "校验窗口中断退出码（rc=$(gen_exit hold)）"
ffprobe -v error "$WD/out/segment_0002.mp4" >/dev/null 2>&1 \
    && ok "校验窗口内已完成的 segment_0002 保留" || bad "校验窗口内已完成的 segment_0002 保留"
grep -q '中断收尾：.*完整，保留' "$WD/gen.log" \
    && ok "打印保留完整片段提示" || bad "打印保留完整片段提示"

printf '%s\n' "== 5. generate_series.sh：失败重试 / 不重试 =="

# -r 1：首段第一次失败，重试成功
WD="$TMP/retry"; mkdir -p "$WD/out"; printf 'fake-png' > "$WD/start.png"
make_mock_h3 "$WD"; cp "$GS" "$WD/"
LOG="$WD/h3.log"; MARK="$WD/fail.marker"
run_gen "$WD" retry 40 \
    MOCK_FAIL_ONCE=1 MOCK_FAIL_MARKER="$MARK" MOCK_LOG="$LOG" \
    MOCK_DELAY_FROM=3 MOCK_DELAY=30 \
    -- -i start.png -d out -p "PROMPT" -r 1 --steps 2 --frames 5
wait_file "$WD/out/segment_0002.mp4" 20 || bad "重试后应连续生成到 0002"
sleep 1
: > "$TMP/retry.trigger"
wait_gen retry 25

[ "$(gen_exit retry)" = "130" ] \
    && ok "重试场景中断退出码 130" || bad "重试场景中断退出码（rc=$(gen_exit retry)）"
grep -q '第 1 次失败：h3 退出码' "$WD/gen.log" \
    && ok "记录第一次失败警告" || bad "记录第一次失败警告"
ffprobe -v error "$WD/out/segment_0001.mp4" >/dev/null 2>&1 \
    && ok "重试后 segment_0001 有效" || bad "重试后 segment_0001 有效"

# -r 0：失败立即中止，不重试，退出码 1
WD="$TMP/noretry"; mkdir -p "$WD/out"; printf 'fake-png' > "$WD/start.png"
make_mock_h3 "$WD"; cp "$GS" "$WD/"
LOG="$WD/h3.log"
run_gen "$WD" noretry 30 \
    MOCK_FAIL_ALWAYS=1 MOCK_LOG="$LOG" \
    -- -i start.png -d out -p "PROMPT" -r 0 --steps 2 --frames 5
wait_gen noretry 15

[ "$(gen_exit noretry)" = "1" ] \
    && ok "-r 0 失败立即中止（rc=$(gen_exit noretry)）" || bad "-r 0 失败立即中止（rc=$(gen_exit noretry)）"
grep -q '中止：' "$WD/gen.log" && ok "打印中止信息" || bad "打印中止信息"
[ -z "$(ls "$WD/out" 2>/dev/null)" ] \
    && ok "失败后无残留片段" || bad "失败后无残留片段"

printf '%s\n' "== 6. generate_series.sh：续跑 + 启动扫描 =="

# 预置损坏 segment_0001 + 有效 segment_0002 → 删除损坏，从 0003 续跑，ref=0002
WD="$TMP/resume"; mkdir -p "$WD/out"; printf 'fake-png' > "$WD/start.png"
make_mock_h3 "$WD"; cp "$GS" "$WD/"
printf 'garbage-not-mp4' > "$WD/out/segment_0001.mp4"
mkseg "$WD/out/segment_0002.mp4"
LOG="$WD/h3.log"
run_gen "$WD" resume 40 \
    MOCK_CORRUPT_FROM=3 MOCK_CORRUPT_DELAY=30 MOCK_LOG="$LOG" \
    -- -i start.png -d out -p "PROMPT" --steps 2 --frames 5
wait_grep "$LOG" 'segment_0003.mp4' 20 || bad "续跑应从 0003 开始"
sleep 1
: > "$TMP/resume.trigger"
wait_gen resume 25

[ "$(gen_exit resume)" = "130" ] \
    && ok "续跑场景中断退出码 130" || bad "续跑场景中断退出码（rc=$(gen_exit resume)）"
grep -q '警告：损坏的片段 segment_0001.mp4' "$WD/gen.log" \
    && ok "启动时警告损坏片段" || bad "启动时警告损坏片段"
[ -f "$WD/out/segment_0001.mp4" ] \
    && bad "损坏片段应被删除" || ok "损坏片段已删除"
grep -q '续跑：最高有效片段编号为 0002' "$WD/gen.log" \
    && ok "打印续跑提示" || bad "打印续跑提示"
grep -q -- '--ref-silent-video out/segment_0002.mp4' "$LOG" \
    && ok "续跑段以最后有效片段为参考" || bad "续跑段以最后有效片段为参考"
[ -f "$WD/out/segment_0003.mp4" ] \
    && bad "被中断的 segment_0003（已写入损坏文件）应被删除" \
    || ok "被中断的 segment_0003（已写入损坏文件）已删除"

printf '%s\n' "== 7. generate_series.sh：EXTRA_ARGS 校验 =="

# --core-reuse 强制 --reuse 1；--seed 透传
WD="$TMP/extra"; mkdir -p "$WD/out"; printf 'fake-png' > "$WD/start.png"
make_mock_h3 "$WD"; cp "$GS" "$WD/"
LOG="$WD/h3.log"
run_gen "$WD" extra 40 \
    MOCK_DELAY_FROM=3 MOCK_DELAY=30 MOCK_LOG="$LOG" \
    -- -i start.png -d out -p "PROMPT" --reuse 2 --core-reuse 4 --seed 7 \
    --steps 2 --frames 5
wait_file "$WD/out/segment_0001.mp4" 20 || bad "--core-reuse 场景 segment_0001 未生成"
sleep 1
: > "$TMP/extra.trigger"
wait_gen extra 25

grep -q '已强制 --reuse 1' "$WD/gen.log" \
    && ok "透传 --core-reuse 时打印强制提示" || bad "透传 --core-reuse 时打印强制提示"
grep -q -- '--reuse 1' "$LOG" \
    && ok "--core-reuse 透传时命令中使用 --reuse 1" || bad "--core-reuse 透传时命令中使用 --reuse 1"
grep -q -- '--seed 7' "$LOG" \
    && ok "--seed 7 原样透传" || bad "--seed 7 原样透传"

# EXTRA_ARGS 中不允许 -o
WD="$TMP/outflag"; mkdir -p "$WD/out"; printf 'fake-png' > "$WD/start.png"
make_mock_h3 "$WD"; cp "$GS" "$WD/"
out="$(cd "$WD" && bash ./generate_series.sh -i start.png -d out -p "PROMPT" -o /tmp/evil.mp4 2>&1)" || true
printf '%s' "$out" | grep -q '不允许出现 -o/--output' \
    && ok "EXTRA_ARGS 含 -o 时报错" || bad "EXTRA_ARGS 含 -o 时报错"

printf '%s\n' "== 8. concat_segments.sh：拼接 =="

WD="$TMP/concat"; mkdir -p "$WD/out"
mkseg "$WD/out/segment_0001.mp4"
mkseg "$WD/out/segment_0002.mp4"
printf 'corrupt-garbage' > "$WD/out/segment_0003.mp4"
mkseg "$WD/out/segment_0004.mp4"
cp "$CS" "$WD/"

outlog="$WD/concat.log"
bash "$CS" "$WD/out" "$WD/final.mp4" > "$outlog" 2>&1
if [ -f "$WD/final.mp4" ]; then
    ok "默认模式拼接成功"
    d="$(dur "$WD/final.mp4")"
    if approx "$d" 0.3 0.08; then
        ok "输出时长约等于 3 个有效片段之和（$d s）"
    else
        bad "输出时长约等于片段之和（$d s）"
    fi
    grep -q '模式：-c copy' "$outlog" && ok "默认模式提示 -c copy" || bad "默认模式提示 -c copy"
    grep -q '警告：跳过损坏的片段' "$outlog" \
        && ok "跳过损坏片段并打印警告" || bad "跳过损坏片段并打印警告"
else
    bad "默认模式拼接成功"
fi

bash "$CS" -r "$WD/out" "$WD/final_r.mp4" > "$outlog" 2>&1
if [ -f "$WD/final_r.mp4" ]; then
    ok "重编码模式拼接成功"
    d="$(dur "$WD/final_r.mp4")"
    approx "$d" 0.3 0.08 \
        && ok "重编码输出时长正确（$d s）" || bad "重编码输出时长正确（$d s）"
else
    bad "重编码模式拼接成功"
fi

# 无有效片段 → 报错
bash "$CS" "$WD/empty" "$WD/empty.mp4" >/dev/null 2>&1
[ "$?" -ne 0 ] && ok "目录不存在时报错" || bad "目录不存在时报错"
mkdir -p "$WD/onlybad"
printf 'garbage' > "$WD/onlybad/segment_0001.mp4"
bash "$CS" "$WD/onlybad" "$WD/bad.mp4" >/dev/null 2>&1
[ "$?" -ne 0 ] && ok "只有损坏片段时报错" || bad "只有损坏片段时报错"

printf '\n通过 %d 项，失败 %d 项\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
