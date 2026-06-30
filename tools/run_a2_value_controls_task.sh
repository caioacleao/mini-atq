#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${ATQ_REPO_DIR:-$PWD}"
OUTPUT_DIR="${OUTPUT_DIR:-results_a2_value_controls_$(date +%Y%m%d)}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

TARGET_SET="${TARGET_SET:-barenco3}"
TARGET_TAG="${TARGET_TAG:-barenco_tof_3}"
TARGET_MANIFEST="${TARGET_MANIFEST:-}"
USE_MANIFEST="${USE_MANIFEST:-0}"
SEEDS="${SEEDS:-2024,2025,2026,2027,2028}"
ARMS="${ARMS:-scalar_mse,scalar_huber_d1,categorical_61,quantile_risk_neutral,quantile_q075}"
TASK_ORDER="${TASK_ORDER:-seed_major}"

BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_MCTS_SIMULATIONS="${NUM_MCTS_SIMULATIONS:-32}"
NUM_TRAINING_STEPS="${NUM_TRAINING_STEPS:-1000}"
EVAL_FREQUENCY_STEPS="${EVAL_FREQUENCY_STEPS:-50}"
CHECKPOINT_FREQUENCY_STEPS="${CHECKPOINT_FREQUENCY_STEPS:-0}"
CHECKPOINT_PAYLOAD="${CHECKPOINT_PAYLOAD:-eval}"
MAX_NUM_MOVES="${MAX_NUM_MOVES:-30}"
GUMBEL_ACTIONS="${GUMBEL_ACTIONS:-8}"
GUMBEL_SCALE="${GUMBEL_SCALE:-0.1}"

REPLAY_CAPACITY="${REPLAY_CAPACITY:-100000}"
REPLAY_MIN_SIZE="${REPLAY_MIN_SIZE:-1000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
NUM_LEARNER_STEPS_PER_ACTOR="${NUM_LEARNER_STEPS_PER_ACTOR:-8}"

EVAL_SEEDS="${EVAL_SEEDS:-0,1,2,3,4}"
EVAL_CONTROLS="${EVAL_CONTROLS:-orbit}"
EVAL_K_VALUES="${EVAL_K_VALUES:-1}"

OVERWRITE="${OVERWRITE:-0}"
RESUME_INCOMPLETE="${RESUME_INCOMPLETE:-0}"
DRY_RUN="${DRY_RUN:-0}"
MODE="${MODE:-run}"
A2_TASK_ID="${A2_TASK_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
JAX_CACHE_DIR="${JAX_CACHE_DIR:-${REPO_DIR}/.jax_cache}"

cd "${REPO_DIR}"

export PYTHONPATH=.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Overridable: 'highest' = full FP32 (deterministic but no TF32 tensor cores, slow on
# Ampere); 'high' = TF32 (much faster, immaterial for the exact GF(2) solve verification).
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-highest}"
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="${JAX_CACHE_DIR}"
mkdir -p "${JAX_CACHE_DIR}"

IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"
IFS=',' read -r -a ARM_ARRAY <<< "${ARMS}"

trim() {
  echo "$1" | xargs
}

# Builds (once) and echoes a per-target manifest path for non-built-in targets
# (e.g. cuccaro_adder_n3, barenco_tof_4) imported from QASM. Mirrors the pattern
# in run_robust_quantile_task.sh so both harnesses share manifest provenance.
ensure_manifest() {
  local target="$1"
  local manifest="${OUTPUT_DIR}/manifests/${target}/targets.json"
  if [[ ! -f "${manifest}" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      mkdir -p "$(dirname "${manifest}")"
      echo "${manifest}"
      return 0
    fi
    "${PYTHON_BIN}" tools/build_benchmark_manifests.py \
      --output_dir "${OUTPUT_DIR}/manifests" \
      --targets "${target}" \
      --seeds "${SEEDS}" \
      --policies gumbel \
      --expected_logs_per_target 1 >/dev/null
  fi
  echo "${manifest}"
}

build_tasks() {
  local arm seed
  case "${TASK_ORDER}" in
    seed_major)
      for seed in "${SEED_ARRAY[@]}"; do
        seed="$(trim "${seed}")"
        [[ -z "${seed}" ]] && continue
        for arm in "${ARM_ARRAY[@]}"; do
          arm="$(trim "${arm}")"
          [[ -z "${arm}" ]] && continue
          echo "${arm}:${seed}"
        done
      done
      ;;
    arm_major)
      for arm in "${ARM_ARRAY[@]}"; do
        arm="$(trim "${arm}")"
        [[ -z "${arm}" ]] && continue
        for seed in "${SEED_ARRAY[@]}"; do
          seed="$(trim "${seed}")"
          [[ -z "${seed}" ]] && continue
          echo "${arm}:${seed}"
        done
      done
      ;;
    *)
      echo "Unknown TASK_ORDER=${TASK_ORDER}" >&2
      return 2
      ;;
  esac
}

task_count() {
  build_tasks | wc -l | tr -d ' '
}

task_by_index() {
  local index="$1"
  build_tasks | sed -n "$((index + 1))p"
}

ensure_provenance() {
  mkdir -p \
    "${OUTPUT_DIR}/logs/${TARGET_TAG}" \
    "${OUTPUT_DIR}/checkpoints/${TARGET_TAG}" \
    "${OUTPUT_DIR}/eval/${TARGET_TAG}" \
    "${OUTPUT_DIR}/analysis" \
    "${OUTPUT_DIR}/provenance"

  if [[ ! -f "${OUTPUT_DIR}/run_config.json" || "${OVERWRITE}" == "1" ]]; then
    cat > "${OUTPUT_DIR}/run_config.json" <<EOF
{
  "stage": "a2_value_controls",
  "target_set": "${TARGET_SET}",
  "target_tag": "${TARGET_TAG}",
  "target_manifest": "${TARGET_MANIFEST}",
  "use_manifest": "${USE_MANIFEST}",
  "seeds": "${SEEDS}",
  "arms": "${ARMS}",
  "task_order": "${TASK_ORDER}",
  "batch_size": ${BATCH_SIZE},
  "num_mcts_simulations": ${NUM_MCTS_SIMULATIONS},
  "num_training_steps": ${NUM_TRAINING_STEPS},
  "eval_frequency_steps": ${EVAL_FREQUENCY_STEPS},
  "checkpoint_frequency_steps": ${CHECKPOINT_FREQUENCY_STEPS},
  "checkpoint_payload": "${CHECKPOINT_PAYLOAD}",
  "max_num_moves": ${MAX_NUM_MOVES},
  "gumbel_actions": ${GUMBEL_ACTIONS},
  "gumbel_scale": ${GUMBEL_SCALE},
  "replay_capacity": ${REPLAY_CAPACITY},
  "replay_min_size": ${REPLAY_MIN_SIZE},
  "train_batch_size": ${TRAIN_BATCH_SIZE},
  "num_learner_steps_per_actor": ${NUM_LEARNER_STEPS_PER_ACTOR},
  "eval_controls": "${EVAL_CONTROLS}",
  "eval_k_values": "${EVAL_K_VALUES}",
  "eval_seeds": "${EVAL_SEEDS}",
  "resume_incomplete": "${RESUME_INCOMPLETE}"
}
EOF
  fi

  git rev-parse HEAD > "${OUTPUT_DIR}/provenance/git_commit.txt" 2>/dev/null || true
  git status --short > "${OUTPUT_DIR}/provenance/git_status_short.txt" 2>/dev/null || true
  git diff --stat > "${OUTPUT_DIR}/provenance/git_diff_stat.txt" 2>/dev/null || true
}

arm_flags() {
  case "$1" in
    scalar_mse)
      echo "--num_value_quantiles=0 --value_risk_quantile=0.0 --value_scalar_loss=mse --value_huber_delta=1.0 --num_value_categorical_bins=0"
      ;;
    scalar_huber_d1)
      echo "--num_value_quantiles=0 --value_risk_quantile=0.0 --value_scalar_loss=huber --value_huber_delta=1.0 --num_value_categorical_bins=0"
      ;;
    scalar_huber_d0.5)
      echo "--num_value_quantiles=0 --value_risk_quantile=0.0 --value_scalar_loss=huber --value_huber_delta=0.5 --num_value_categorical_bins=0"
      ;;
    scalar_huber_d2)
      echo "--num_value_quantiles=0 --value_risk_quantile=0.0 --value_scalar_loss=huber --value_huber_delta=2.0 --num_value_categorical_bins=0"
      ;;
    scalar_huber_d5)
      echo "--num_value_quantiles=0 --value_risk_quantile=0.0 --value_scalar_loss=huber --value_huber_delta=5.0 --num_value_categorical_bins=0"
      ;;
    categorical_61)
      echo "--num_value_quantiles=0 --value_risk_quantile=0.0 --value_scalar_loss=mse --value_huber_delta=1.0 --num_value_categorical_bins=61 --value_support_min=-60.0 --value_support_max=0.0"
      ;;
    categorical_wide)
      # Support covers the realized return range on barenco_tof_3 (returns reach ~-136),
      # removing the [-60,0] clipping confound that handicapped categorical_61.
      echo "--num_value_quantiles=0 --value_risk_quantile=0.0 --value_scalar_loss=mse --value_huber_delta=1.0 --num_value_categorical_bins=161 --value_support_min=-160.0 --value_support_max=0.0"
      ;;
    categorical_81)
      echo "--num_value_quantiles=0 --value_risk_quantile=0.0 --value_scalar_loss=mse --value_huber_delta=1.0 --num_value_categorical_bins=81 --value_support_min=-80.0 --value_support_max=0.0"
      ;;
    categorical_121)
      echo "--num_value_quantiles=0 --value_risk_quantile=0.0 --value_scalar_loss=mse --value_huber_delta=1.0 --num_value_categorical_bins=121 --value_support_min=-120.0 --value_support_max=0.0"
      ;;
    quantile_risk_neutral)
      echo "--num_value_quantiles=8 --value_risk_quantile=0.0 --value_scalar_loss=mse --value_huber_delta=1.0 --num_value_categorical_bins=0"
      ;;
    quantile_q075)
      echo "--num_value_quantiles=8 --value_risk_quantile=0.75 --value_scalar_loss=mse --value_huber_delta=1.0 --num_value_categorical_bins=0"
      ;;
    *)
      echo "Unknown arm: $1" >&2
      return 2
      ;;
  esac
}

is_log_complete() {
  local log_path="$1"
  [[ -f "${log_path}" ]] || return 1
  grep -q "Step: ${NUM_TRAINING_STEPS}" "${log_path}"
}

find_checkpoint() {
  local checkpoint_dir="$1"
  local seed="$2"
  find "${checkpoint_dir}" \
    -name "checkpoint_target-*_seed${seed}_step${NUM_TRAINING_STEPS}.pkl" \
    -print 2>/dev/null | sort | tail -n 1 || true
}

find_latest_full_checkpoint() {
  local checkpoint_dir="$1"
  local seed="$2"
  local checkpoint
  while IFS= read -r checkpoint; do
    if [[ -f "${checkpoint}.json" ]] &&
        grep -q '"checkpoint_payload": "full"' "${checkpoint}.json"; then
      echo "${checkpoint}"
      return 0
    fi
  done < <(
    find "${checkpoint_dir}" \
      -name "checkpoint_target-*_seed${seed}_step*.pkl" \
      -print 2>/dev/null \
      | sed -E 's/.*_step([0-9]+)\.pkl$/\1 &/' \
      | sort -nr \
      | cut -d' ' -f2-
  )
}

run_eval() {
  local checkpoint="$1"
  local eval_csv="$2"
  if [[ "${OVERWRITE}" != "1" && -f "${eval_csv}" ]]; then
    echo "Skipping existing eval CSV: ${eval_csv}"
    return 0
  fi
  local eval_cmd=(
    "${PYTHON_BIN}" "-u" "-m" "alphatensor_quantum.src.demo.evaluate_checkpoint"
    "--checkpoint=${checkpoint}"
    "--output_csv=${eval_csv}"
    "--controls=${EVAL_CONTROLS}"
    "--k_values=${EVAL_K_VALUES}"
    "--eval_seeds=${EVAL_SEEDS}"
  )
  echo "${eval_cmd[*]}"
  echo "  -> ${eval_csv}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${eval_cmd[@]}"
  fi
}

run_task() {
  local task="$1"
  local arm seed
  IFS=':' read -r arm seed <<< "${task}"
  ensure_provenance

  local target_flag
  if [[ -n "${TARGET_MANIFEST}" ]]; then
    target_flag="--target_manifest=${TARGET_MANIFEST}"
  elif [[ "${USE_MANIFEST}" == "1" ]]; then
    target_flag="--target_manifest=$(ensure_manifest "${TARGET_TAG}")"
  else
    target_flag="--target_set=${TARGET_SET}"
  fi

  local run_tag="arm-${arm}_target-${TARGET_TAG}_seed${seed}"
  local log_dir="${OUTPUT_DIR}/logs/${TARGET_TAG}/arm-${arm}"
  local checkpoint_dir="${OUTPUT_DIR}/checkpoints/${TARGET_TAG}/arm-${arm}/seed${seed}"
  local eval_csv="${OUTPUT_DIR}/eval/${TARGET_TAG}/eval_${run_tag}.csv"
  local log_path="${log_dir}/${run_tag}.log"
  mkdir -p "${log_dir}" "${checkpoint_dir}" "$(dirname "${eval_csv}")"

  local checkpoint
  if [[ "${OVERWRITE}" != "1" ]] &&
      is_log_complete "${log_path}" &&
      checkpoint="$(find_checkpoint "${checkpoint_dir}" "${seed}")" &&
      [[ -n "${checkpoint}" ]]; then
    echo "Skipping complete train log: ${log_path}"
  else
    read -r -a extra_flags <<< "$(arm_flags "${arm}")"
    local resume_checkpoint=""
    if [[ "${RESUME_INCOMPLETE}" == "1" ]]; then
      checkpoint="$(find_latest_full_checkpoint "${checkpoint_dir}" "${seed}")"
      if [[ -n "${checkpoint}" ]]; then
        resume_checkpoint="${checkpoint}"
      fi
    fi
    local cmd=(
      "${PYTHON_BIN}" "-u" "-m" "alphatensor_quantum.src.demo.run_demo"
      "--use_gadgets=false"
      "${target_flag}"
      "--search_policy=gumbel"
      "--gumbel_max_num_considered_actions=${GUMBEL_ACTIONS}"
      "--gumbel_scale=${GUMBEL_SCALE}"
      "--seed=${seed}"
      "--batch_size=${BATCH_SIZE}"
      "--num_mcts_simulations=${NUM_MCTS_SIMULATIONS}"
      "--num_training_steps=${NUM_TRAINING_STEPS}"
      "--eval_frequency_steps=${EVAL_FREQUENCY_STEPS}"
      "--max_num_moves=${MAX_NUM_MOVES}"
      "--replay_capacity=${REPLAY_CAPACITY}"
      "--replay_min_size=${REPLAY_MIN_SIZE}"
      "--train_batch_size=${TRAIN_BATCH_SIZE}"
      "--actor_rollout_length=${MAX_NUM_MOVES}"
      "--num_learner_steps_per_actor=${NUM_LEARNER_STEPS_PER_ACTOR}"
      "--value_target_mode=mc_return"
      "--checkpoint_dir=${checkpoint_dir}"
      "--checkpoint_frequency_steps=${CHECKPOINT_FREQUENCY_STEPS}"
      "--checkpoint_payload=${CHECKPOINT_PAYLOAD}"
      ${ALGEBRAIC_PRIOR_MODE:+"--algebraic_prior_mode=${ALGEBRAIC_PRIOR_MODE}"}
      ${ALGEBRAIC_PRIOR_BETA:+"--algebraic_prior_beta=${ALGEBRAIC_PRIOR_BETA}"}
      ${VALUE_TARGET_TRANSFORM:+"--value_target_transform=${VALUE_TARGET_TRANSFORM}"}
      "${extra_flags[@]}"
    )
    if [[ -n "${resume_checkpoint}" ]]; then
      cmd+=("--resume_checkpoint=${resume_checkpoint}")
    fi
    echo "${cmd[*]}"
    echo "  -> ${log_path}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      "${cmd[@]}" > "${log_path}" 2>&1
    fi
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  checkpoint="$(find_checkpoint "${checkpoint_dir}" "${seed}")"
  if [[ -z "${checkpoint}" ]]; then
    echo "Missing checkpoint for task=${task}" >&2
    return 1
  fi
  run_eval "${checkpoint}" "${eval_csv}"
}

case "${MODE}" in
  count)
    task_count
    ;;
  print)
    total="$(task_count)"
    echo "total_tasks=${total}"
    idx=0
    while [[ "${idx}" -lt "${total}" ]]; do
      task="$(task_by_index "${idx}")"
      echo "===== task_id=${idx} task=${task} ====="
      DRY_RUN=1 run_task "${task}"
      idx=$((idx + 1))
    done
    ;;
  run)
    task="$(task_by_index "${A2_TASK_ID}")"
    if [[ -z "${task}" ]]; then
      echo "No task for A2_TASK_ID=${A2_TASK_ID}" >&2
      exit 2
    fi
    echo "task_id=${A2_TASK_ID} task=${task}"
    run_task "${task}"
    ;;
  *)
    echo "Unknown MODE=${MODE}" >&2
    exit 2
    ;;
esac
