#!/usr/bin/env bash
#
# generate_series.sh — 从起始图片持续生成 15 秒视频片段，循环不断，Ctrl+C 停止。
# 首段用 --first-frame 锚定起始图；后续段用 --ref-silent-video 参考上一段实现运动延续。
#
# 用法：
#   ./generate_series.sh -i START.png -d OUTDIR -p "PROMPT" [选项...] [h3 透传参数...]
#
# 选项：
#   -i IMG        起始图（必填；仅首段 --first-frame 使用）
#   -d DIR        输出目录（必填；自动创建）
#   -p TEXT       首段提示词（必填）
#   -c TEXT       延续段提示词（默认 "Continue the motion in this clip."）
#   -w N / -h N   宽 / 高（默认 512 / 512）
#   -s N          去噪步数（默认 50）
#   --frames N    每段帧数（默认 362；合法档位 5+17n）
#   --layers N    DiT 层数（默认 50）
#   --reuse N     denoiser reuse（默认 1）
#   -r N          失败重试次数（默认 1；0 关闭重试）
#   --help        显示本帮助
#
# 未识别的参数会原样透传给 h3（EXTRA_ARGS）。
# 注意：EXTRA_ARGS 中不应出现 -o/--output（输出路径由本脚本管理）；
#       透传 --core-reuse 时本脚本强制 --reuse 1（二者在 h3 中互斥）。
#
# 退出码：0 正常结束；1 生成失败中止（已完成片段保留）；130 被 Ctrl+C 中断。
#
# 依赖：./h3、ffmpeg、ffprobe（零外部依赖，仅 bash 内建 + 上述工具）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H3="$SCRIPT_DIR/h3"
MODEL_DIR="$SCRIPT_DIR/MiniMax-H3"
CONCAT="$SCRIPT_DIR/concat_segments.sh"

IMG=""
DIR=""
FIRST_PROMPT=""
CONT_PROMPT="Continue the motion in this clip."
W=512
H=512
STEPS=50
F=362
L=50
R=1
RETRY=1
EXTRA_ARGS=()

usage() {
    cat >&2 <<'EOF'
用法：./generate_series.sh -i START.png -d OUTDIR -p "PROMPT" [选项...] [h3 透传参数...]

选项：
  -i IMG        起始图（必填；仅首段 --first-frame 使用）
  -d DIR        输出目录（必填；自动创建）
  -p TEXT       首段提示词（必填）
  -c TEXT       延续段提示词（默认 "Continue the motion in this clip."）
  -w N / -h N   宽 / 高（默认 512 / 512）
  -s N          去噪步数（默认 50）
  --frames N    每段帧数（默认 362；合法档位 5+17n）
  --layers N    DiT 层数（默认 50）
  --reuse N     denoiser reuse（默认 1）
  -r N          失败重试次数（默认 1；0 关闭重试）
  --help        显示本帮助
EOF
}

die() { echo "错误：$*" >&2; exit 1; }

fail_usage() {
    echo "错误：$*" >&2
    usage
    exit 1
}

is_digits() { case "$1" in ''|*[!0-9]*) return 1;; *) return 0;; esac; }

# ---- 手写参数解析：支持 -k value、-k=value、--key value、--key=value ----
while [ "$#" -gt 0 ]; do
    case "$1" in
        -i) [ "$#" -ge 2 ] || die "选项 -i 缺少参数"; IMG="$2"; shift 2 ;;
        -i=*) IMG="${1#-i=}"; shift ;;
        -d) [ "$#" -ge 2 ] || die "选项 -d 缺少参数"; DIR="$2"; shift 2 ;;
        -d=*) DIR="${1#-d=}"; shift ;;
        -p) [ "$#" -ge 2 ] || die "选项 -p 缺少参数"; FIRST_PROMPT="$2"; shift 2 ;;
        -p=*) FIRST_PROMPT="${1#-p=}"; shift ;;
        -c) [ "$#" -ge 2 ] || die "选项 -c 缺少参数"; CONT_PROMPT="$2"; shift 2 ;;
        -c=*) CONT_PROMPT="${1#-c=}"; shift ;;
        -w) [ "$#" -ge 2 ] || die "选项 -w 缺少参数"; W="$2"; shift 2 ;;
        -w=*) W="${1#-w=}"; shift ;;
        -h) [ "$#" -ge 2 ] || die "选项 -h 缺少参数"; H="$2"; shift 2 ;;
        -h=*) H="${1#-h=}"; shift ;;
        -s) [ "$#" -ge 2 ] || die "选项 -s 缺少参数"; STEPS="$2"; shift 2 ;;
        -s=*) STEPS="${1#-s=}"; shift ;;
        -r) [ "$#" -ge 2 ] || die "选项 -r 缺少参数"; RETRY="$2"; shift 2 ;;
        -r=*) RETRY="${1#-r=}"; shift ;;
        --frames) [ "$#" -ge 2 ] || die "选项 --frames 缺少参数"; F="$2"; shift 2 ;;
        --frames=*) F="${1#--frames=}"; shift ;;
        --layers) [ "$#" -ge 2 ] || die "选项 --layers 缺少参数"; L="$2"; shift 2 ;;
        --layers=*) L="${1#--layers=}"; shift ;;
        --reuse) [ "$#" -ge 2 ] || die "选项 --reuse 缺少参数"; R="$2"; shift 2 ;;
        --reuse=*) R="${1#--reuse=}"; shift ;;
        --help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ---- 校验必填项与数值 ----
[ -n "$IMG" ] || fail_usage "缺少必填参数 -i（起始图）"
[ -n "$DIR" ] || fail_usage "缺少必填参数 -d（输出目录）"
[ -n "$FIRST_PROMPT" ] || fail_usage "缺少必填参数 -p（首段提示词）"
[ -f "$IMG" ] || die "起始图不存在：$IMG"

for v in "$W" "$H" "$STEPS" "$F" "$L" "$R"; do
    is_digits "$v" || die "非法数值：$v"
done
is_digits "$RETRY" || die "非法重试次数：$RETRY"

# ---- EXTRA_ARGS 校验：不允许 -o/--output；--core-reuse 强制 --reuse 1 ----
i=0
while [ "$i" -lt "${#EXTRA_ARGS[@]}" ]; do
    case "${EXTRA_ARGS[$i]}" in
        -o|-o=*|-o?*|--output|--output=*)
            die "EXTRA_ARGS 中不允许出现 -o/--output（输出路径由脚本管理）" ;;
        --core-reuse|--core-reuse=*)
            R=1
            echo "注意：检测到 --core-reuse 透传，已强制 --reuse 1（二者在 h3 中互斥）" ;;
    esac
    i=$((i+1))
done

[ -x "$H3" ] || die "找不到可执行的 h3：$H3"
[ -d "$MODEL_DIR" ] || die "模型目录不存在：$MODEL_DIR"

mkdir -p "$DIR"

# ---- 中断处理：置标志 + 兜底 kill 当前 h3 子进程 ----
INTERRUPTED=0
H3_PID=""
trap 'INTERRUPTED=1; [ -n "${H3_PID:-}" ] && kill "$H3_PID" 2>/dev/null' INT

# ---- 续跑：校验既有 segment_*.mp4，损坏删除并警告，next=最大有效编号+1 ----
max=0
for seg in "$DIR"/segment_*.mp4; do
    [ -e "$seg" ] || continue
    base="$(basename "$seg")"
    num="${base#segment_}"
    num="${num%.mp4}"
    if ! is_digits "$num"; then
        echo "警告：无法识别编号的片段 ${base}，已删除"
        rm -f "$seg"
        continue
    fi
    if ffprobe -v error "$seg" >/dev/null 2>&1; then
        [ "$num" -gt "$max" ] && max="$num"
    else
        echo "警告：损坏的片段 ${base}，已删除"
        rm -f "$seg"
    fi
done
next=$((max + 1))
if [ "$max" -gt 0 ]; then
    echo "续跑：最高有效片段编号为 ${max}，将从 segment_$(printf %04d "$next").mp4 继续"
fi

# ---- 主循环：持续生成，Ctrl+C 停止 ----
while [ "$INTERRUPTED" -ne 1 ]; do
    if [ "$next" -eq 1 ]; then
        ref_args=(--first-frame "$IMG")
        prompt="$FIRST_PROMPT"
        echo "[1] prompt=${prompt}（预计约 30.5 分钟：加载 30s + 生成 ~30min）"
    else
        ref_args=(--ref-silent-video "$DIR/segment_$(printf %04d $((next-1))).mp4")
        prompt="$CONT_PROMPT"
        echo "[$next] prompt=${prompt}（预计约 32 分钟：加载 30s + 参考编码 82s + 生成 ~30min）"
    fi
    out="$DIR/segment_$(printf %04d "$next").mp4"
    rc=1
    for ((attempt=0; attempt<=RETRY; attempt++)); do
        [ "$INTERRUPTED" -eq 1 ] && break          # h3 启动前查中断
        echo "[$next] 第 $((attempt+1))/$((RETRY+1)) 次尝试：h3 生成中..."
        "$H3" -d "$MODEL_DIR" -p "$prompt" --width "$W" --height "$H" \
            --frames "$F" --steps "$STEPS" --layers "$L" --reuse "$R" \
            "${ref_args[@]}" -o "$out" "${EXTRA_ARGS[@]}" &
        H3_PID="$!"
        if wait "$H3_PID"; then
            rc=0
        else
            rc=$?
        fi
        H3_PID=""
        [ "$INTERRUPTED" -eq 1 ] && break          # Ctrl+C：不再重试
        if [ "$rc" -eq 0 ] && ffprobe -v error "$out" >/dev/null 2>&1; then
            echo "[$next] 完成：$out"
            break
        fi
        [ "$INTERRUPTED" -eq 1 ] && break          # 校验窗口被中断：不 rm，交由收尾判定
        rm -f "$out"
        if [ "$rc" -ne 0 ]; then
            echo "[$next] 第 $((attempt+1)) 次失败：h3 退出码 $rc"
        else
            echo "[$next] 第 $((attempt+1)) 次失败：ffprobe 校验不通过"
        fi
        rc=1
    done
    [ "$INTERRUPTED" -eq 1 ] && break              # 中断收尾，不进下一段
    if [ "$rc" -ne 0 ]; then
        echo "中止：segment_$(printf %04d "$next") 失败，已完成 $((next-1)) 段"
        exit 1
    fi
    next=$((next+1))
done

# ---- Ctrl+C 收尾：校验当前段，清理不完整文件，打印保留片段与拼接提示 ----
cur="$DIR/segment_$(printf %04d "$next").mp4"
if [ -f "$cur" ]; then
    if ffprobe -v error "$cur" >/dev/null 2>&1; then
        echo "中断收尾：$cur 完整，保留"
    else
        echo "中断收尾：删除不完整片段 $cur"
        rm -f "$cur"
    fi
fi
echo
echo "已中断。当前保留的有效片段："
n=0
for seg in "$DIR"/segment_*.mp4; do
    [ -e "$seg" ] || continue
    if ffprobe -v error "$seg" >/dev/null 2>&1; then
        echo "  $seg"
        n=$((n+1))
    else
        echo "  （忽略损坏）$seg"
    fi
done
[ "$n" -eq 0 ] && echo "  （无）"
echo
echo "拼接所有片段为一个视频："
echo "  $CONCAT \"$DIR\" \"$DIR/final.mp4\""
exit 130
