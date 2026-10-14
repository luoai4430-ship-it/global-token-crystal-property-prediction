#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/TextResidualSHFMat
PY=/root/autodl-tmp/micromamba/envs/sftmat/bin/python
RUNROOT=/root/autodl-tmp/TextResidualSHFMat/matched_ablation_runs
CACHE=/root/matched_token_cache
LOG=/root/matched_ablation_queue.log

exec > >(tee -a "$LOG") 2>&1
echo "QUEUE_START $(date -Is)"

wait_existing() {
  while pgrep -f 'matched_text_ablation_runner.py' >/dev/null 2>&1; do
    echo "WAIT_EXISTING $(date -Is)"
    sleep 30
  done
}

run_group() {
  local task="$1" cache="$2"
  local variants=(CLS MaskedMean CosineAttention)
  local pending=()
  for variant in "${variants[@]}"; do
    if [[ -f "$RUNROOT/$task/$variant/summary.json" ]]; then
      echo "SKIP_COMPLETE $task $variant $(date -Is)"
    else
      pending+=("$variant")
    fi
  done
  for ((offset=0; offset<${#pending[@]}; offset+=2)); do
    local pids=()
    for variant in "${pending[@]:offset:2}"; do
      local out="$RUNROOT/$task/$variant"
      rm -rf "$out"
      mkdir -p "$out"
      nohup "$PY" matched_text_ablation_runner.py \
        --task "$task" --variant "$variant" \
        --token_cache "$cache" --output_dir "$out" \
        </dev/null > "$out/run.log" 2>&1 &
      pids+=("$!")
      echo "START $task $variant pid=${pids[-1]} $(date -Is)"
    done
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
    echo "DONE_BATCH $task offset=$offset $(date -Is)"
  done
  echo "DONE_GROUP $task $(date -Is)"
}

generate_cache() {
  local name="$1" input="$2" idcol="$3" textcol="$4"
  rm -rf "$CACHE/$name"
  mkdir -p "$CACHE/$name"
  "$PY" scripts/cache_matbert_token_features.py \
    --input "$input" --output_dir "$CACHE/$name" \
    --model_name /root/autodl-tmp/matbert-base-cased \
    --id_column "$idcol" --text_column "$textcol" \
    --max_length 128 --batch_size 64 --device cuda \
    --progress_every 4096 > "$CACHE/${name}.log" 2>&1
  echo "CACHE_READY $name $(date -Is)"
}

# The three MP-Bandgap runs were started interactively before this queue.
wait_existing
run_group MP-FormationEnergy /root/matched_token_cache/mp_shared
rm -rf /root/matched_token_cache/mp_shared /root/matched_token_cache/mp_shared.log

generate_cache j_mbj matched_ablation_inputs/Jarvis-Bandgap_MBJ.csv Id Description
run_group Jarvis-Bandgap_MBJ /root/matched_token_cache/j_mbj
rm -rf /root/matched_token_cache/j_mbj /root/matched_token_cache/j_mbj.log

generate_cache j_fe matched_ablation_inputs/Jarvis-FormationEnergy.csv Id Description
run_group Jarvis-FormationEnergy /root/matched_token_cache/j_fe
rm -rf /root/matched_token_cache/j_fe /root/matched_token_cache/j_fe.log

generate_cache j_bulk matched_ablation_inputs/Jarvis-BulkModulusKv.csv Id Description
run_group Jarvis-BulkModulusKv /root/matched_token_cache/j_bulk
rm -rf /root/matched_token_cache/j_bulk /root/matched_token_cache/j_bulk.log

generate_cache mp_shear matched_ablation_inputs/MP-ShearModuli.csv id text
run_group MP-ShearModuli /root/matched_token_cache/mp_shear
rm -rf /root/matched_token_cache/mp_shear /root/matched_token_cache/mp_shear.log

date -Is > /root/matched_ablation_queue.done
echo "QUEUE_DONE $(date -Is)"
