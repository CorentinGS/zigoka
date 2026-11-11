#!/usr/bin/env bash
# One unattended round of the 19×19 student (docs/designs/moka-19.md):
#   1. a KataGo self-play corpus in data/<name>/, in chunks of one KataGo process each (its memory
#      grows with its NN cache, so no process runs for hours);
#   2. the combined recipe (weight decay 0.1, teacher-only value target, warm start from Moka v2's
#      trunk) on the pilot corpus plus the new one, once per seed;
#   3. scores on the pilot's test set, and the export of the seed with the best validation loss to
#      ../models/local/<name>/, built into zig-out/local/;
#   4. games at 1 s per move against KataGo's rank_5k human profile.
# Run nothing else on the GPU meanwhile: KataGo processes next to a training run fault the Radeon.
# Every finished step leaves a marker in checkpoints/<name>/, so running it again resumes.
#
# usage (from anywhere): pipeline/scripts/round-19.sh <name>
# env, with defaults: CHUNKS=10 CHUNK_POSITIONS=250000 FIRST_SEED=11 SEEDS="1 2 3" EPOCHS=4 GAMES=200
#   KATAGO_NETS=~/.cache/zigoka-katago/nets
# At the pilot's speeds: 5 h 30 of self-play, 45 min of training per seed, 1 h 50 of games.
set -euo pipefail

name=${1:?usage: pipeline/scripts/round-19.sh <name>}
if [[ ! $name =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  echo "<name> is the model id, such as moka-19-r1" >&2
  exit 2
fi

cd "$(dirname "$0")/.."
chunks=${CHUNKS:-10}
chunk_positions=${CHUNK_POSITIONS:-250000}
first_seed=${FIRST_SEED:-11}
seeds=${SEEDS:-1 2 3}
epochs=${EPOCHS:-4}
games=${GAMES:-200}
nets=${KATAGO_NETS:-$HOME/.cache/zigoka-katago/nets}
teacher=$nets/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz
human=$nets/b18c384nbt-humanv0.bin.gz
run=checkpoints/$name
python=.venv/bin/python

mkdir -p "$run"
log() { echo "$(date '+%F %T') $*" | tee -a "$run/round.log"; }
# The ROCm libraries print "(null)" lines; keep them out of the logs.
quiet() { grep --line-buffered -v "(null)" || true; }

# The pilot's seeds are 2 (corpus), 3 (test) and 4 (validation): FIRST_SEED stays clear of them.
for ((chunk = 0; chunk < chunks; chunk++)); do
  [[ -e $run/chunk-$chunk.done ]] && continue
  seed=$((first_seed + chunk))
  log "self-play, chunk $((chunk + 1)) of $chunks, seed $seed"
  "$python" -m go_pipeline.selfplay --katago katago/katago-rocm --model "$teacher" --out "data/$name" \
    --board-size 19 --positions "$chunk_positions" --games-in-flight 64 --shard-rows 5000 --seed "$seed" \
    2>&1 | quiet >> "$run/selfplay.log"
  touch "$run/chunk-$chunk.done"
done

checkpoints=()
for seed in $seeds; do
  checkpoints+=("$run/s$seed.pt")
  [[ -e $run/s$seed.done ]] && continue
  log "training, seed $seed, $epochs epochs"
  "$python" -m go_pipeline.train --data data/pilot-19 "data/$name" --val-data data/pilot-19-val \
    --init-dir ../vendor/moka-v2/model --batch-size 256 --epochs "$epochs" --non-deterministic \
    --weight-decay 0.1 --outcome-mix 0 --seed "$seed" --out "$run/s$seed.pt" \
    2>&1 | quiet > "$run/train-s$seed.log"
  touch "$run/s$seed.done"
done

if [[ ! -e $run/scores.done ]]; then
  log "scores on the pilot's test set"
  "$python" scripts/score.py data/pilot-19-test "${checkpoints[@]}" 2>&1 | quiet > "$run/scores.log"
  touch "$run/scores.done"
fi

best=$(for seed in $seeds; do
  echo "$(sed -n 's/.*best val int8 loss \([0-9.]*\)).*/\1/p' "$run/train-s$seed.log") $seed"
done | sort -g | head -1 | cut -d' ' -f2)

if [[ ! -e $run/export.done ]]; then
  log "export of seed $best to models/local/$name"
  "$python" -m go_pipeline.export --checkpoint "$run/s$best.pt" --output-dir "../models/local/$name" \
    > "$run/export.log" 2>&1
  (cd .. && zig build models wasm search -Dlocal-model="$name:synthetic-19-v2" && pnpm build:js) \
    >> "$run/export.log" 2>&1
  touch "$run/export.done"
fi

if [[ ! -e $run/match.done ]]; then
  log "$games games against rank_5k at 1 s per move"
  (cd .. && KATAGO_BINARY=$PWD/pipeline/katago/katago-rocm KATAGO_MODEL=$teacher KATAGO_HUMAN_MODEL=$human \
    node match/dist/match-19.js --first zig-reuse --first-model "$name" --first-time 1000 --second katago \
    --katago-profile rank_5k --games "$games" --parallel 6 2>> "pipeline/$run/match.err") > "$run/match.log"
  touch "$run/match.done"
fi

log "done: seed $best exported, scores in $run/scores.log, games in $run/match.log"
cat "$run/scores.log" "$run/match.log"
