#!/usr/bin/env bash
# Adapts FaceFusion to the post-processing contract in docs/POSTPROCESSING.md:
#
#   faceswap-facefusion.sh --input IN.mp4 --output OUT.mp4
#
# The contract carries only the video, so the face to put in comes from the
# environment. Both variables are written into .env by install.sh, except the
# source face, which only you can choose:
#
#   H3_FACEFUSION_DIR   the FaceFusion checkout
#   H3_FACESWAP_SOURCE  an image of the face to use
#   H3_FACEFUSION_PYTHON  optional, the interpreter of its environment
#
# Do not run this on a real person who has not agreed to it.
#
# FaceFusion's own command line is its own, and it changes between releases:
# this wrapper is written against the 3.x `headless-run`. If your version
# names things differently, this file is the one place to adjust.

set -euo pipefail

input=""
output=""
while [ $# -gt 0 ]; do
    case "$1" in
        --input) input="${2:-}"; shift 2;;
        --output) output="${2:-}"; shift 2;;
        *) printf 'faceswap: unknown argument: %s\n' "$1" >&2; exit 2;;
    esac
done

[ -n "$input" ] && [ -n "$output" ] ||
    { printf 'faceswap: --input and --output are both required\n' >&2; exit 2; }
: "${H3_FACEFUSION_DIR:?set it to the FaceFusion checkout}"
: "${H3_FACESWAP_SOURCE:?set it to an image of the face to use}"
[ -f "$H3_FACESWAP_SOURCE" ] ||
    { printf 'faceswap: no such source image: %s\n' "$H3_FACESWAP_SOURCE" >&2; exit 2; }

python="${H3_FACEFUSION_PYTHON:-python3}"
exec "$python" "$H3_FACEFUSION_DIR/facefusion.py" headless-run \
    --processors face_swapper \
    --source-paths "$H3_FACESWAP_SOURCE" \
    --target-path "$input" \
    --output-path "$output"
