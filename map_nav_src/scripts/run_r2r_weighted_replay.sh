#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_r2r_weighted_replay.sh collect|train|eval|diagnose
# Run with the same architecture flags as your baseline checkpoint.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${BASE_CKPT:?Set BASE_CKPT to your fine-tuned DUET checkpoint}"
DATA_ROOT="${DATA_ROOT:-../datasets}"
OUT_DIR="${OUT_DIR:-../outputs/r2r-weighted-replay}"
TRAJECTORIES="${TRAJECTORIES:-${OUT_DIR}/train-k4.jsonl}"
STUDENT_CKPT="${STUDENT_CKPT:-${OUT_DIR}/student/ckpts/best_val_unseen}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

common=(
  --root_dir "$DATA_ROOT" --dataset r2r --world_size 1 --seed "${SEED:-0}"
  --tokenizer bert --enc_full_graph --graph_sprels --fusion dynamic
  --expert_policy spl --train_alg dagger
  --num_l_layers 9 --num_x_layers 4 --num_pano_layers 2
  --max_action_len 15 --max_instr_len 200 --batch_size "${BATCH_SIZE:-8}"
  --features vitbase --image_feat_size 768 --angle_feat_size 4
  --feat_dropout 0.4 --dropout 0.5 --optim adamW --lr 1e-5 --ml_weight 0.2
)

case "${1:-}" in
  collect)
    python -m r2r.collect_trajectories "${common[@]}" \
      --output_dir "$OUT_DIR/collection" --resume_file "$BASE_CKPT" \
      --trajectory_file "$TRAJECTORIES" --num_samples "${K:-4}" \
      --collect_split train --collect_limit "${LIMIT:-0}"
    ;;
  train)
    python -m r2r.main_nav "${common[@]}" \
      --output_dir "$OUT_DIR/student" --resume_file "$BASE_CKPT" \
      --replay_file "$TRAJECTORIES" --replay_beta "${BETA:-1.0}" \
      --replay_weight "${REPLAY_WEIGHT:-0.2}" --policy_stop \
      --iters "${ITERS:-10000}" --log_every "${LOG_EVERY:-1000}" --eval_first
    ;;
  eval)
    # Same STOP protocol for both models; each model executes one greedy path.
    python -m r2r.main_nav "${common[@]}" \
      --output_dir "$OUT_DIR/eval-base" --resume_file "$BASE_CKPT" \
      --policy_stop --test --submit
    python -m r2r.main_nav "${common[@]}" \
      --output_dir "$OUT_DIR/eval-student" --resume_file "$STUDENT_CKPT" \
      --policy_stop --test --submit
    ;;
  diagnose)
    python -m r2r.collect_trajectories "${common[@]}" \
      --output_dir "$OUT_DIR/diagnostic" --resume_file "$BASE_CKPT" \
      --trajectory_file "$OUT_DIR/val-unseen-diagnostic.jsonl" \
      --collect_split val_unseen --num_samples "${K:-4}" --collect_limit "${LIMIT:-100}"
    ;;
  *) echo 'Usage: run_r2r_weighted_replay.sh collect|train|eval|diagnose' >&2; exit 2 ;;
esac
