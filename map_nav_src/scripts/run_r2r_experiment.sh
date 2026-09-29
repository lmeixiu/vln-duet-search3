#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$repo_root/experiments/variant.env"
cd "$repo_root/map_nav_src"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
mode="${1:-help}"
if [[ "$mode" == help ]]; then
  echo 'Usage: run_r2r_experiment.sh collect|train|eval-base|eval-best|eval-last|diagnose'
  echo "Branch default variant: $EXPERIMENT_VARIANT (dagger, uniform, reward)"
  echo 'Set BASE_CKPT, DATA_ROOT, OUT_ROOT; DRY_RUN=1 prints commands without running.'
  exit 0
fi
: "${BASE_CKPT:?Set BASE_CKPT to the original fine-tuned DUET checkpoint}"
DATA_ROOT="${DATA_ROOT:-$repo_root/datasets}"
OUT_ROOT="${OUT_ROOT:-$repo_root/outputs/ablation-v2}"
SEED="${SEED:-0}"
COLLECT_SEED="${COLLECT_SEED:-0}"
K="${K:-4}"
RUN_ID="${RUN_ID:-pilot}"
RUN_DIR="$OUT_ROOT/$RUN_ID/$EXPERIMENT_VARIANT/seed-$SEED"
TRAJECTORIES="${TRAJECTORIES:-$OUT_ROOT/shared/train-k${K}-seed${COLLECT_SEED}.jsonl}"
common=(
  --root_dir "$DATA_ROOT" --dataset r2r --world_size 1 --seed "$SEED"
  --tokenizer bert --enc_full_graph --graph_sprels --fusion dynamic
  --expert_policy spl --train_alg dagger --num_l_layers 9 --num_x_layers 4 --num_pano_layers 2
  --max_action_len 15 --max_instr_len 200 --batch_size "${BATCH_SIZE:-8}"
  --features vitbase --image_feat_size 768 --angle_feat_size 4
  --feat_dropout 0.4 --dropout 0.5 --optim adamW --lr "${LR:-1e-5}" --ml_weight 0.2
)
checkpoint="$BASE_CKPT"
case "$mode" in
  collect|diagnose)
    split=train
    trajectory="$TRAJECTORIES"
    limit="${LIMIT:-0}"
    if [[ "$mode" == diagnose ]]; then
      split=val_unseen
      trajectory="${DIAGNOSTIC_FILE:-$OUT_ROOT/shared/val-unseen-k${K}-seed${COLLECT_SEED}.jsonl}"
      limit="${LIMIT:-100}"
    fi
    run_output="${trajectory}.run"
    cmd=(python -u -m r2r.collect_trajectories "${common[@]}"
      --seed "$COLLECT_SEED" --output_dir "$run_output" --resume_file "$BASE_CKPT"
      --trajectory_file "$trajectory" --num_samples "$K" --collect_split "$split" --collect_limit "$limit"
      --reward_success "${REWARD_SUCCESS:-1}" --reward_ndtw "${REWARD_NDTW:-1}" --reward_spl "${REWARD_SPL:-1}")
    ;;
  train)
    run_output="$RUN_DIR/student"
    cmd=(python -u -m r2r.main_nav "${common[@]}" --output_dir "$run_output" --resume_file "$BASE_CKPT"
      --policy_stop --iters "${ITERS:-5000}" --log_every "${LOG_EVERY:-1000}" --eval_first)
    case "$EXPERIMENT_VARIANT" in
      dagger) ;;
      uniform|reward)
        cmd+=(--replay_file "$TRAJECTORIES" --replay_mode "$EXPERIMENT_VARIANT"
          --replay_beta "${BETA:-1}" --replay_weight "${REPLAY_WEIGHT:-0.2}") ;;
      *) echo 'Invalid experiment variant' >&2; exit 2 ;;
    esac
    ;;
  eval-base|eval-best|eval-last)
    if [[ "$mode" == eval-base ]]; then
      run_output="$OUT_ROOT/$RUN_ID/baseline/seed-$SEED/${EVAL_TAG:-validation}"
    else
      checkpoint="$RUN_DIR/student/ckpts/best_val_unseen"
      [[ "$mode" == eval-last ]] && checkpoint="$RUN_DIR/student/ckpts/latest_dict"
      checkpoint="${STUDENT_CKPT:-$checkpoint}"
      run_output="$RUN_DIR/$mode/${EVAL_TAG:-validation}"
    fi
    cmd=(python -u -m r2r.main_nav "${common[@]}" --output_dir "$run_output" --resume_file "$checkpoint"
      --policy_stop --test --save_predictions --eval_splits val_train_seen val_seen val_unseen)
    ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
printf 'Variant=%s mode=%s output=%s\n' "$EXPERIMENT_VARIANT" "$mode" "$run_output"
printf '%q ' "${cmd[@]}"; printf '\n'
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0
[[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 2; }
[[ -d "$DATA_ROOT" ]] || { echo "Missing data directory: $DATA_ROOT" >&2; exit 2; }
if [[ "$mode" == train && "$EXPERIMENT_VARIANT" != dagger && ! -f "$TRAJECTORIES" ]]; then
  echo "Missing replay file: $TRAJECTORIES" >&2; exit 2
fi
# Fail before model loading rather than silently mixing old/new predictions or checkpoints.
[[ ! -e "$run_output" ]] || { echo 'Output already exists. Choose a new RUN_ID, EVAL_TAG or trajectory path.' >&2; exit 2; }
mkdir -p "$run_output"
printf '%q ' "${cmd[@]}" > "$run_output/command.txt"; printf '\n' >> "$run_output/command.txt"
git -C "$repo_root" rev-parse HEAD > "$run_output/commit.txt"
git -C "$repo_root" status --porcelain > "$run_output/worktree_status.txt"
sha256sum "$checkpoint" > "$run_output/checkpoint.sha256"
if [[ "$mode" == train && "$EXPERIMENT_VARIANT" != dagger ]]; then
  sha256sum "$TRAJECTORIES" > "$run_output/trajectories.sha256"
fi
start_seconds=$SECONDS
"${cmd[@]}" 2>&1 | tee "$run_output/console.log"
printf '%s\n' "$((SECONDS - start_seconds))" > "$run_output/run_wall_seconds.txt"
