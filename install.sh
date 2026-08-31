#!/usr/bin/env bash
# install.sh — prepares a machine to run h3c studio.
#
# It checks the prerequisites, puts the repository in place and writes a .env.
# By default it downloads nothing: the 465 GB checkpoint and the optional
# face-swapping runtime are asked for, one at a time, and can be declined.
#
# Read this file before running it. It is deliberately not written to be piped
# from a URL into a shell: fetch it, read it, then run it.
#
#   ./install.sh --dir ~/h3            # repository and .env only
#   ./install.sh --dir ~/h3 --with-model
#   ./install.sh --yes                 # no questions, defaults, no downloads
#
# MIT licensed, like the rest of this repository.

set -euo pipefail

REPO_URL="https://github.com/matrixfede/h3.c.git"
REPO_BRANCH=""
MODEL_REPO="MiniMaxAI/MiniMax-H3"
# The optional face-swapping runtime. Its models are its own business: it
# fetches them itself, on first use, from its own sources.
FACEFUSION_URL="https://github.com/facefusion/facefusion.git"
# The checkpoint is about 465 GB; leave room for the images and a few videos.
MODEL_GIB=480
WORK_GIB=20

DEST=""
WANT_MODEL=""
WANT_FACESWAP=""
ASSUME_YES=0

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
info() { printf '  %s\n' "$*"; }
warn() { printf '  ! %s\n' "$*" >&2; }
die()  { printf '\n! %s\n' "$*" >&2; exit 1; }

usage() {
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --dir DIR         where the repository goes (default: the current checkout,
                    otherwise ./h3.c)
  --repo URL        repository to clone from (default: the one above)
  --branch NAME     branch to clone (default: whatever the remote's is)
  --with-model      download the MiniMax-H3 checkpoint (about 465 GB)
  --without-model   do not download it, and do not ask
  --with-faceswap   install the optional face-swapping runtime
  --without-faceswap
                    do not install it, and do not ask
  --yes             never ask; anything not requested with a flag is skipped
  -h, --help        this text
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dir) DEST="${2:-}"; shift 2 || die "--dir needs a directory";;
        --dir=*) DEST="${1#*=}"; shift;;
        --repo) REPO_URL="${2:-}"; shift 2 || die "--repo needs a URL";;
        --repo=*) REPO_URL="${1#*=}"; shift;;
        --branch) REPO_BRANCH="${2:-}"; shift 2 || die "--branch needs a name";;
        --branch=*) REPO_BRANCH="${1#*=}"; shift;;
        --with-model) WANT_MODEL=yes; shift;;
        --without-model) WANT_MODEL=no; shift;;
        --with-faceswap) WANT_FACESWAP=yes; shift;;
        --without-faceswap) WANT_FACESWAP=no; shift;;
        --yes|-y) ASSUME_YES=1; shift;;
        -h|--help) usage; exit 0;;
        *) usage >&2; die "unknown option: $1";;
    esac
done

# A question only where there is someone to answer it. Everywhere else the
# answer is no, because the expensive choices are the optional ones.
ask() {
    local question="$1"
    if [ "$ASSUME_YES" = 1 ] || [ ! -t 0 ]; then return 1; fi
    local reply=""
    read -r -p "  $question [y/N] " reply || return 1
    case "$reply" in [yY]|[yY][eE][sS]) return 0;; *) return 1;; esac
}

have() { command -v "$1" >/dev/null 2>&1; }

free_gib() {
    # Space on the filesystem that will hold this path, existing or not.
    local path="$1"
    while [ ! -d "$path" ] && [ "$path" != "/" ]; do path=$(dirname "$path"); done
    df -PB1G "$path" | awk 'NR == 2 { print $4 }'
}

check_prerequisites() {
    say "Checking what this machine already has"
    local missing=()
    have git || missing+=("git — to fetch the repository")
    # Writing the .env below is a python3 heredoc: needed on every path.
    have python3 || missing+=("python3 — to write the .env")

    # Two ways to build and run: the containers, or the compiler on the host.
    if have docker; then
        info "docker: yes"
        # ffmpeg and ICU live in the images; the host does not need them.
        info "ffmpeg/ICU: not required on the host — the images carry them"
    elif have nvcc && have make; then
        info "docker: no, but nvcc and make are here — the local path works"
        have ffmpeg || missing+=("ffmpeg — h3 muxes video and audio with it")
        { have pkg-config && pkg-config --modversion icu-uc >/dev/null 2>&1; } ||
            missing+=("pkg-config + libicu-dev — h3 links ICU (README: 72+)")
    else
        missing+=("docker, or nvcc and make — one of the two ways to build h3")
    fi
    have git && info "git: yes"
    have python3 && info "python3: yes"

    if [ ${#missing[@]} -gt 0 ]; then
        printf '\n' >&2
        for item in "${missing[@]}"; do warn "missing: $item"; done
        die "install what is missing above, then run this again."
    fi
}

require_space() {
    local path="$1" needed="$2" what="$3" free
    free=$(free_gib "$path")
    info "free space for $what: ${free} GiB (about ${needed} GiB needed)"
    [ "$free" -ge "$needed" ] ||
        die "not enough room for $what: ${free} GiB free, about ${needed} GiB needed."
}

# Where the repository is, or is going to be. Running this from inside a
# checkout uses that checkout: no second copy of the same thing.
resolve_destination() {
    local here=""
    here=$(cd -- "$(dirname -- "$0")" && pwd)
    if [ -z "$DEST" ] && [ -f "$here/docker-compose.yml" ] && [ -f "$here/h3.c" ]; then
        DEST="$here"
    fi
    DEST="${DEST:-$PWD/h3.c}"
    mkdir -p -- "$(dirname -- "$DEST")"
}

fetch_repository() {
    if [ -f "$DEST/docker-compose.yml" ] && [ -f "$DEST/h3.c" ]; then
        say "Repository"
        info "already at $DEST — left as it is"
        return
    fi
    [ ! -e "$DEST" ] || [ -z "$(ls -A -- "$DEST" 2>/dev/null)" ] ||
        die "$DEST exists and is not an h3.c checkout. Pick another --dir."
    say "Fetching the repository into $DEST"
    if [ -n "$REPO_BRANCH" ]; then
        git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "$DEST"
    else
        git clone --depth 1 "$REPO_URL" "$DEST"
    fi
}

# A clone of the wrong branch is the likeliest way to end up here with a
# directory that looks right and has no Studio in it. Say so, instead of
# failing later on a missing file.
verify_checkout() {
    local missing=""
    for needed in .env.example docker-compose.yml webui/backend/app/main.py; do
        [ -e "$DEST/$needed" ] || missing="$missing $needed"
    done
    [ -z "$missing" ] || die "$DEST has no web UI in it (missing:$missing).
  That branch of the repository does not carry it. Clone one that does:
      $0 --dir $DEST --branch <branch-with-the-web-ui>"
}

# The checkpoint. Around 465 GB, hours of download, and the one thing without
# which nothing can be generated — so it is asked for explicitly and can be
# declined. The commands are the ones the README documents, unchanged.
model_present() {
    [ -f "$1/model_index.json" ] && [ -d "$1/FL2VA" ] && [ -d "$1/Ref2VA" ]
}

download_model() {
    local dir="$1"
    if model_present "$dir"; then
        info "already complete in $dir — not downloading it again"
        return
    fi
    have hf || die "the Hugging Face CLI is missing. Install it with:
      pip install -U \"huggingface_hub[cli]\"
  then run this again, or download the checkpoint yourself into $dir."
    require_space "$dir" "$MODEL_GIB" "the checkpoint"
    info "about 465 GB from $MODEL_REPO — this takes hours and resumes if cut"
    info "a gated or private repository needs 'hf auth login' first"
    hf download "$MODEL_REPO" --local-dir "$dir"
    info "checking that every file arrived"
    hf cache verify "$MODEL_REPO" --local-dir "$dir" --fail-on-missing-files
}

# Writes or replaces one line of the .env, leaving everything else alone.
set_env_var() {
    python3 - "$DEST/.env" "$1" "$2" <<'ENVPY'
import sys
path, key, value = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(path).read().splitlines(True)
for index, line in enumerate(lines):
    if line.startswith(f"{key}="):
        lines[index] = f"{key}={value}\n"
        break
else:
    lines.append(f"{key}={value}\n")
open(path, "w").writelines(lines)
ENVPY
}

# The optional face-swapping runtime. Off unless asked for: nothing is
# installed by a default answer, and the licence and consent notice is printed
# before anything is fetched.
faceswap_notice() {
    info "FaceFusion is a separate project, with its own licence, and it"
    info "downloads its own models the first time it runs. The face-swapping"
    info "models known to us are licensed for non-commercial or research use"
    info "only: checking what you install is your responsibility."
    info "Never use face replacement on a real person who has not agreed to it."
}

install_faceswap() {
    local dir="$DEST/vendor/facefusion"
    if [ -f "$dir/facefusion.py" ]; then
        info "already installed in $dir — left as it is"
    else
        say "Fetching FaceFusion into $dir"
        mkdir -p -- "$(dirname -- "$dir")"
        git clone --depth 1 "$FACEFUSION_URL" "$dir"
        info "running its own installer — this pulls its Python dependencies"
        ( cd "$dir" && python3 install.py --onnxruntime default --skip-conda )
    fi
    set_env_var H3_FACEFUSION_DIR "$dir"
    set_env_var H3_FACESWAP_CMD "$DEST/scripts/faceswap-facefusion.sh"
    info "wrote H3_FACESWAP_CMD into $DEST/.env"
    info "one thing is still yours to set: H3_FACESWAP_SOURCE, the image of"
    info "the face to use. Until it is set, the stage refuses to run."
}

# The .env is the user's file: written once, never overwritten.
write_env() {
    local env="$DEST/.env" model_dir="$1"
    say "Configuration"
    if [ -f "$env" ]; then
        info ".env already exists — left as it is"
        return
    fi
    cp -- "$DEST/.env.example" "$env"
    # The value can contain slashes, so the substitution is not a s|..| one.
    python3 - "$env" "$model_dir" <<'PY'
import sys
path, model = sys.argv[1], sys.argv[2]
lines = []
for line in open(path).read().splitlines(True):
    if line.startswith("H3_MODEL_DIR="):
        line = f"H3_MODEL_DIR={model}\n"
    lines.append(line)
open(path, "w").writelines(lines)
PY
    info "wrote $env with H3_MODEL_DIR=$model_dir"
}

main() {
    resolve_destination
    check_prerequisites
    require_space "$DEST" "$WORK_GIB" "the build and the videos"
    fetch_repository
    verify_checkout

    local model_dir="$DEST/MiniMax-H3"
    write_env "$model_dir"

    say "The MiniMax-H3 checkpoint"
    if [ -z "$WANT_MODEL" ]; then
        info "about 465 GB. Nothing can be generated without it, but it can"
        info "be fetched later: ./install.sh --dir $DEST --with-model"
        if ask "Download it now?"; then WANT_MODEL=yes; else WANT_MODEL=no; fi
    fi
    if [ "$WANT_MODEL" = yes ]; then
        download_model "$model_dir"
    else
        info "skipped — fetch it later with: ./install.sh --dir $DEST --with-model"
    fi

    say "Optional: replacing faces in the finished video"
    faceswap_notice
    if [ -z "$WANT_FACESWAP" ]; then
        if ask "Install FaceFusion now?"; then
            WANT_FACESWAP=yes
        else
            WANT_FACESWAP=no
        fi
    fi
    if [ "$WANT_FACESWAP" = yes ]; then
        install_faceswap
    else
        info "face replacement: skipped, and nothing was downloaded"
    fi

    say "Done"
    info "next: cd $DEST && docker compose up --build"
    info "then open http://127.0.0.1:8080"
}

main "$@"
