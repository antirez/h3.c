#!/usr/bin/env bash
#
# concat_segments.sh — 按文件名自然排序拼接目录中的 segment_*.mp4 片段为单个视频。
#
# 用法：
#   ./concat_segments.sh [选项] <片段目录> <输出.mp4>
#
# 选项：
#   -r, --reencode  重编码兜底（libx264 + AAC）；默认 -c copy 无损拼接
#   --help          显示本帮助
#
# 行为：按字典序（4 位零填充命名 = 自然序）收集片段；逐段 ffprobe 校验，
#       损坏/缺失的片段跳过并打印警告；成功打印输出路径与总时长，
#       失败删除不完整输出并以非零退出。
#
# 依赖：ffmpeg、ffprobe（零外部依赖，仅 bash 内建 + 上述工具）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REENCODE=0
DIR=""
OUT=""
list=""
trap 'rm -f "$list"' EXIT

usage() {
    cat >&2 <<'EOF'
用法：./concat_segments.sh [选项] <片段目录> <输出.mp4>

选项：
  -r, --reencode  重编码兜底（libx264 + AAC）；默认 -c copy 无损拼接
  --help          显示本帮助
EOF
}

die() { echo "错误：$*" >&2; exit 1; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        -r|--reencode) REENCODE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) die "未知选项：$1" ;;
        *)
            if [ -z "$DIR" ]; then
                DIR="$1"
            elif [ -z "$OUT" ]; then
                OUT="$1"
            else
                die "多余的位置参数：$1"
            fi
            shift ;;
    esac
done

[ -n "$DIR" ] || die "缺少 <片段目录> 参数"
[ -n "$OUT" ] || die "缺少 <输出.mp4> 参数"
[ -d "$DIR" ] || die "片段目录不存在：$DIR"

segments=()
for seg in "$DIR"/segment_*.mp4; do
    [ -e "$seg" ] || continue
    if ffprobe -v error "$seg" >/dev/null 2>&1; then
        segments+=("$seg")
    else
        echo "警告：跳过损坏的片段 $seg"
    fi
done

if [ "${#segments[@]}" -eq 0 ]; then
    die "目录中没有有效的 segment_*.mp4 片段"
fi

echo "共 ${#segments[@]} 个有效片段，开始拼接："
for seg in "${segments[@]}"; do
    echo "  $seg"
done

# 转义路径供 ffconcat 脚本解析（ffmpeg av_get_token 规则：\X → X，
# 空格是 token 终止符需转义；不用单引号包裹，否则引号段内的转义会失效）。
# 逐字符处理，避免 bash 参数展开中反斜杠字面量语义的版本差异。
escape_path() {
    local s="$1" out="" c
    while [ -n "$s" ]; do
        c="${s%"${s#?}"}"
        case "$c" in
            \\ ) out="${out}\\\\" ;;   # \ → \\
            \' ) out="${out}\'" ;;      # ' → \'
            \  ) out="${out}\ " ;;      # 空格 → \<space>
            *  ) out="${out}$c" ;;
        esac
        s="${s#?}"
    done
    printf '%s' "$out"
}

# concat list：绝对路径 + 转义。
# 列表文件放在输出目录内（隐藏点文件，$$ 保证唯一），避免 /tmp 下被
# 预置符号链接截断（TOCTOU）；退出时由 trap 清理。
list="$DIR/.concat_list_$$.txt"
: > "$list"
for seg in "${segments[@]}"; do
    abspath="$(cd "$(dirname "$seg")" && pwd)/$(basename "$seg")"
    printf "file %s\n" "$(escape_path "$abspath")" >> "$list"
done

if [ "$REENCODE" -eq 1 ]; then
    echo "模式：重编码（libx264 + AAC）"
    if ! ffmpeg -y -v error -f concat -safe 0 -i "$list" \
        -c:v libx264 -preset medium -crf 18 -c:a aac -b:a 128k "$OUT"; then
        rm -f "$OUT"
        die "ffmpeg 拼接失败，已删除不完整输出 $OUT"
    fi
else
    echo "模式：-c copy 无损拼接"
    if ! ffmpeg -y -v error -f concat -safe 0 -i "$list" -c copy "$OUT"; then
        rm -f "$OUT"
        die "ffmpeg 拼接失败，已删除不完整输出 $OUT"
    fi
fi

dur="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT" 2>/dev/null || true)"
echo "完成：$OUT"
echo "总时长：${dur:-未知} 秒"
