#!/usr/bin/env bash
# verify.sh — gate unico di validazione per agenti di codice.
# Uso: ./scripts/verify.sh [all|static|unit|integration]
# Esito: ultima riga "VERIFY: PASS" oppure "VERIFY: FAIL" + exit code.
#
# Rileva automaticamente lo stack presente. Output volutamente compatto:
# il consumatore è un agente, non un umano — ogni riga superflua è token bruciati.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

MODE="${1:-all}"
FAIL=0
mkdir -p logs/agent

step() { printf '\n── %s\n' "$1"; }
has()  { command -v "$1" >/dev/null 2>&1; }
run()  { echo "  \$ $*"; "$@" || FAIL=1; }

# Web UI: stack Python isolato nel proprio venv, non su PATH.
WEBUI_VENV="webui/backend/.venv/bin"
[[ -x "$WEBUI_VENV/pytest" ]] && WEBUI=1 || WEBUI=0

# Rilevamento stack
PY=0; JS=0
[[ -f pyproject.toml || -f setup.py || -f requirements.txt || -n "$(ls -1 ./*.py 2>/dev/null)" ]] && PY=1
[[ -f package.json ]] && JS=1

# ─────────────────────────────── ANALISI STATICA ───────────────────────────────
if [[ "$MODE" == "static" || "$MODE" == "all" ]]; then
  step "Analisi statica"
  if [[ $PY -eq 1 ]]; then
    has ruff  && run ruff check . --output-format=concise
    has mypy  && run mypy . --no-error-summary --pretty=False
  fi
  if [[ $JS -eq 1 ]]; then
    has npx && [[ -f tsconfig.json ]] && run npx --no-install tsc --noEmit --pretty false
    has npx && run npx --no-install eslint . --format=compact
  fi
  if [[ $WEBUI -eq 1 ]]; then
    run "$WEBUI_VENV/ruff" check webui
  fi
  # Debug print dimenticati.
  # Il pattern è spezzato ("AGENT""DBG") così questo script non matcha se stesso;
  # esclusi anche .md e scripts/ per non segnalare la documentazione del pack.
  DBG=$(grep -rn --exclude-dir={.git,node_modules,logs,.venv,scripts,MiniMax-H3} \
        --exclude="*.md" "AGENT""DBG|" . 2>/dev/null | head -5)
  if [[ -n "$DBG" ]]; then
    echo "  ! debug print temporanei ancora presenti — rimuovere prima di chiudere il task"
    echo "$DBG" | sed 's/^/    /'
    FAIL=1
  fi
fi

# ──────────────────────────────── TEST UNITARI ────────────────────────────────
if [[ "$MODE" == "unit" || "$MODE" == "all" ]]; then
  step "Test unitari"
  if [[ "$(uname -s)" == "Linux" && -f Makefile ]] &&
     grep -q '^cuda-runtime-test:' Makefile; then
    run make PLATFORM=Linux host-portable-test
    run make PLATFORM=Linux tokenizer-portable-test
    if [[ -f MiniMax-H3/FL2VA/audio_vae/model.safetensors &&
          -f MiniMax-H3/Ref2VA/video_vae/source/model.safetensors ]]; then
      run make PLATFORM=Linux checkpoint-schema-test
    else
      echo "  · checkpoint MiniMax-H3 assente: smoke schema non applicabile"
    fi
    if has nvcc; then
      run make PLATFORM=Linux cuda-runtime-test
      run make PLATFORM=Linux cuda-primitives-test
      run make PLATFORM=Linux cuda-rope-tokens-test
      run make PLATFORM=Linux cuda-linear-test
      run make PLATFORM=Linux cuda-attention-test
      run make PLATFORM=Linux cuda-ops-test
      run make PLATFORM=Linux test
    else
      echo "  ! nvcc assente: impossibile eseguire il gate CUDA"
      FAIL=1
    fi
  fi
  if [[ $WEBUI -eq 1 ]]; then
    run "$WEBUI_VENV/pytest" -q -x --tb=short -m "not integration" webui/backend/tests
  fi
  if [[ $PY -eq 1 ]] && has pytest; then
    run pytest -q -x --tb=short -m "not integration"
  fi
  if [[ $JS -eq 1 ]] && has npx; then
    if grep -q '"vitest"' package.json 2>/dev/null; then
      run npx --no-install vitest run --reporter=dot
    elif grep -q '"jest"' package.json 2>/dev/null; then
      run npx --no-install jest --silent
    fi
  fi
fi

# ───────────────────────────── TEST DI INTEGRAZIONE ────────────────────────────
if [[ "$MODE" == "integration" || "$MODE" == "all" ]]; then
  step "Test di integrazione"
  if [[ $PY -eq 1 ]] && has pytest; then
    run pytest -q --tb=short -m integration || true   # nessun test marcato = non è un errore
  fi
fi

# ──────────────────────────────────── ESITO ───────────────────────────────────
if [[ $FAIL -eq 0 ]]; then
  RESULT="VERIFY: PASS"
else
  RESULT="VERIFY: FAIL"
fi
echo "$RESULT ($(date +%H:%M:%S), mode=$MODE)" | tee logs/agent/last_verify.txt
exit $FAIL
