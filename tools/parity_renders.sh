#!/bin/zsh
# parity_renders.sh — fixed-seed 1 s renders of each models.json profile with two
# h3 binaries, compared by ffmpeg framemd5 of the decoded video and audio.
#
#   tools/parity_renders.sh BIN_A BIN_B OUT_DIR PROFILE...
#
# Each profile is rendered the way gen.sh renders it (registry run dir + the
# profile's env map), identically for both binaries. ref2va profiles get --ref-image,
# fl2va profiles --first-frame, both from $REF with the prompt in $PROMPT_FILE.
# Env knobs: ROOT (checkout holding models.json and server/registry.py; required),
# REF, PROMPT_FILE, SEED (101), FRAMES (22), SIZE (768x1024), STEPS (profile "steps"
# or 6), EXTRA_ENV (KEY=VALUE ... applied to both binaries, e.g. "H3_SPARSE_ATTN=0").
# The per-pair int8 DiT cache under models/dit-cache is used only if it already
# exists; it is never written. Exit status 1 if any profile differs or fails.
# Run only when no other h3 render is in flight (check `pgrep -fl h3`).
set -uo pipefail
(( $# >= 4 )) || { echo "usage: $0 BIN_A BIN_B OUT_DIR PROFILE..." >&2; exit 2; }
bin_a=${1:A} bin_b=${2:A} out=${3:A}
shift 3
root=${ROOT:?set ROOT to the checkout holding models.json and server/registry.py}
ref=${REF:-$root/test-inputs/ref_woman.png}
prompt_file=${PROMPT_FILE:-$root/test-inputs/ref_woman.txt}
seed=${SEED:-101} frames=${FRAMES:-22} size=${SIZE:-768x1024}
extra=(${=EXTRA_ENV:-})
mkdir -p $out
summary=$out/summary.tsv
[[ -s $summary ]] || print "profile\tframes\tlatent\tframes_a\tframes_b\tseconds_a\tseconds_b" > $summary
failures=0

for profile in "$@"; do
  pipeline=$(python3 -I -c '
import json, sys
reg = json.load(open(sys.argv[1]))
print({f["id"]: f["pipeline"] for f in reg["folds"]}[sys.argv[2]])
' $root/models.json $profile) || { print "FAIL $profile: not in models.json"; failures=$((failures + 1)); continue; }
  kind=$([[ $pipeline == ref2va ]] && echo refs || echo text)
  dir=$(python3 -I $root/server/registry.py resolve $kind $profile) || { print "FAIL $profile: resolve"; failures=$((failures + 1)); continue; }
  sampler=(${=$(python3 -I $root/server/registry.py env $kind $profile)})
  steps=${STEPS:-$(python3 -I -c '
import json, sys
reg = json.load(open(sys.argv[1]))
print({f["id"]: f.get("steps", 6) for f in reg["folds"]}[sys.argv[2]])
' $root/models.json $profile)}
  inputs=($([[ $pipeline == ref2va ]] && echo --ref-image || echo --first-frame) $ref)
  pair=${dir:t}
  cache=$root/models/dit-cache/$pair.int8
  cache_env=()
  [[ -e $cache.$pipeline ]] && cache_env=(H3_DIT_INT8_CACHE=$cache)
  results=()
  for label in a b; do
    bin=$([[ $label == a ]] && echo $bin_a || echo $bin_b)
    clip=$out/${profile}_$label.mp4
    log=$out/${profile}_$label.log
    t0=$SECONDS
    env H3_VIDEO_SHIFT=6 H3_VAE_PRECISION=bf16 H3_VAE_CACHE=$root/models/vae-cache \
      H3_VAE_PREFETCH=1 H3_DUMP_VIDEO_LATENT=$out/${profile}_$label.latent \
      $sampler $cache_env $extra \
      $bin -d $dir -p "$(<$prompt_file)" $inputs --width ${size%x*} --height ${size#*x} \
      --frames $frames --steps $steps --seed $seed -o $clip </dev/null >$log 2>&1
    rc=$?
    results+=($((SECONDS - t0)))
    if (( rc != 0 )) || [[ ! -s $clip ]]; then
      print "FAIL $profile ($label) rc=$rc, see $log"
      failures=$((failures + 1))
      continue 2
    fi
    ffmpeg -v error -nostdin -i $clip -f framemd5 $out/${profile}_$label.framemd5 \
      || { print "FAIL $profile ($label): framemd5"; failures=$((failures + 1)); continue 2; }
  done
  fa=$(grep -vc '^#' $out/${profile}_a.framemd5)
  fb=$(grep -vc '^#' $out/${profile}_b.framemd5)
  if cmp -s $out/${profile}_a.framemd5 $out/${profile}_b.framemd5; then
    verdict=identical
  else
    verdict=DIFFERENT
    failures=$((failures + 1))
  fi
  latent=$(cmp -s $out/${profile}_a.latent $out/${profile}_b.latent && echo identical || echo DIFFERENT)
  print "$profile frames=$verdict latent=$latent ($fa/$fb frame hashes; ${results[1]} s / ${results[2]} s)"
  print "$profile\t$verdict\t$latent\t$fa\t$fb\t${results[1]}\t${results[2]}" >> $summary
done
(( failures == 0 ))
