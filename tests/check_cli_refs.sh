#!/bin/sh
# Reference-count limits of the CLI. Usage: check_cli_refs.sh H3 [IMAGE_DIR]
h3=$1
image_dir=$2

expect() {
    want_status=$1; want_text=$2; shift 2
    output=$("$@" 2>&1)
    status=$?
    case "$output" in *"$want_text"*) found=1 ;; *) found=0 ;; esac
    if [ "$status" -ne "$want_status" ] || [ "$found" -ne 1 ]; then
        echo "FAIL: expected exit $want_status and \"$want_text\", got exit $status: $output" >&2
        exit 1
    fi
}

args=""
n=0
while [ $n -lt 13 ]; do args="$args --ref-image x.png"; n=$((n + 1)); done
# Parse time, before the model is known: the video limit and its message.
expect 2 "Ref2VA supports at most 12 references" "$h3" -d /nonexistent -p p $args

if [ -n "$image_dir" ] && [ -f "$image_dir/transformer_diffusers/config.json" ]; then
    args=""
    n=0
    while [ $n -lt 9 ]; do args="$args --ref-image x.png"; n=$((n + 1)); done
    expect 2 "at most 8 --ref-image options" "$h3" -d "$image_dir" -p p $args
    # --no-cache is an image option: it parses, so the next check speaks.
    expect 2 "the interactive session is video-only" "$h3" -d "$image_dir" --no-cache
else
    echo "skip: image model directory is not installed"
fi
echo "ok: CLI reference limits"
