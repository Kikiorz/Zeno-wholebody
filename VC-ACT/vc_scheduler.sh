#!/usr/bin/env bash
# vc_scheduler.sh — VC-ACT 训练调度器
# 自动管理 4 GPU，训练 12 个 ACT + 2 个分类器
#
# 用法:
#   nohup bash /workspace/Mult-skill\ ACT/VC-ACT/vc_scheduler.sh \
#     &> /workspace/Mult-skill\ ACT/VC-ACT/scheduler.log &
#
# 训练计划 (14 个任务):
#   ┌─────────┬──────────────────────────────────┬──────────┐
#   │  任务ID │            描述                   │  类型    │
#   ├─────────┼──────────────────────────────────┼──────────┤
#   │  k10_c0 │ vc_k10 cluster0 ACT (6 eps)      │ ACT      │
#   │  k10_c1 │ vc_k10 cluster1 ACT (16 eps)     │ ACT      │
#   │  k10_c2 │ vc_k10 cluster2 ACT (14 eps)     │ ACT      │
#   │  k10_c3 │ vc_k10 cluster3 ACT (5 eps)      │ ACT      │
#   │  k10_c4 │ vc_k10 cluster4 ACT (9 eps)      │ ACT      │
#   │  k10_c5 │ vc_k10 cluster5 ACT (4 eps)      │ ACT      │
#   │  k10_c6 │ vc_k10 cluster6 ACT (5 eps)      │ ACT      │
#   │  k10_c7 │ vc_k10 cluster7 ACT (15 eps)     │ ACT      │
#   │  k10_c8 │ vc_k10 cluster8 ACT (11 eps)     │ ACT      │
#   │  k10_c9 │ vc_k10 cluster9 ACT (15 eps)     │ ACT      │
#   │ auto_c0 │ vc_auto cluster0 ACT (62 eps)    │ ACT      │
#   │ auto_c1 │ vc_auto cluster1 ACT (38 eps)    │ ACT      │
#   │ cls_k10 │ 4-cam cluster classifier k=10    │ 分类器   │
#   │ cls_auto│ 4-cam cluster classifier k=2     │ 分类器   │
#   └─────────┴──────────────────────────────────┴──────────┘
#
# 调度策略:
#   - 轮询检查 GPU 0-3，通过 nvidia-smi 检测实际占用
#   - 哪个 GPU 空闲就分配任务，动态利用所有可用 GPU
#   - 分类器训练很快，排在最后

set -uo pipefail

PROJECT_ROOT="/workspace/Mult-skill ACT"
VC_ROOT="${PROJECT_ROOT}/VC-ACT"
TRAIN_SCRIPT="${PROJECT_ROOT}/train/run_train.py"
PYTHON="/venv/mult-act/bin/python3"
LOG_DIR="${VC_ROOT}/logs"
mkdir -p "${LOG_DIR}"

WANDB_KEY="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"

# ============================================================
# Task definitions
# ============================================================
# Format: "task_id|type|config_or_args"
# type: act (use run_train.py) or classifier (use python script)
TASKS=(
    "k10_c0|act|${VC_ROOT}/config/exp_k10/cluster0.yaml"
    "k10_c1|act|${VC_ROOT}/config/exp_k10/cluster1.yaml"
    "k10_c2|act|${VC_ROOT}/config/exp_k10/cluster2.yaml"
    "k10_c3|act|${VC_ROOT}/config/exp_k10/cluster3.yaml"
    "k10_c4|act|${VC_ROOT}/config/exp_k10/cluster4.yaml"
    "k10_c5|act|${VC_ROOT}/config/exp_k10/cluster5.yaml"
    "k10_c6|act|${VC_ROOT}/config/exp_k10/cluster6.yaml"
    "k10_c7|act|${VC_ROOT}/config/exp_k10/cluster7.yaml"
    "k10_c8|act|${VC_ROOT}/config/exp_k10/cluster8.yaml"
    "k10_c9|act|${VC_ROOT}/config/exp_k10/cluster9.yaml"
    "auto_c0|act|${VC_ROOT}/config/exp_auto/cluster0.yaml"
    "auto_c1|act|${VC_ROOT}/config/exp_auto/cluster1.yaml"
    "cls_k10|classifier|k10"
    "cls_auto|classifier|auto"
)

# GPU tracking — all 4 GPUs, dynamically check availability
GPUS=(0 1 2 3)
declare -A GPU_PID GPU_TASK
for gpu_id in "${GPUS[@]}"; do
    GPU_PID[${gpu_id}]=0
    GPU_TASK[${gpu_id}]="idle"
done

TASK_IDX=0

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ============================================================
# Check if a GPU is occupied by an external process (not ours)
# ============================================================
gpu_has_external_process() {
    local gpu_id=$1
    # Query nvidia-smi for PIDs using this GPU
    local pids
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${gpu_id}" 2>/dev/null | tr -d ' ')
    if [ -z "$pids" ]; then
        return 1  # no processes -> not occupied
    fi
    # Check if any of these PIDs are NOT one of our tracked tasks
    local our_pid="${GPU_PID[${gpu_id}]}"
    while IFS= read -r pid; do
        [ -z "$pid" ] && continue
        if [ "$pid" != "$our_pid" ]; then
            return 0  # external process found
        fi
    done <<< "$pids"
    return 1  # only our own process
}

# ============================================================
# Check if auto_c1 data is ready
# ============================================================
is_auto_c1_ready() {
    local info="${PROJECT_ROOT}/data/vc_auto/cluster1_v30/meta/info.json"
    if [ -f "$info" ]; then
        # Check if finalized (has total_episodes > 0)
        local eps
        eps=$($PYTHON -c "import json; print(json.load(open('${info}'))['total_episodes'])" 2>/dev/null)
        if [ -n "$eps" ] && [ "$eps" -gt 0 ]; then
            return 0
        fi
    fi
    return 1
}

# ============================================================
# Launch a task on a given GPU
# ============================================================
launch_task() {
    local gpu_id=$1
    local task_str=$2
    local task_id config_or_args task_type

    IFS='|' read -r task_id task_type config_or_args <<< "$task_str"

    # Skip auto_c1 if data not ready
    if [ "$task_id" == "auto_c1" ] && ! is_auto_c1_ready; then
        log "SKIP auto_c1: data not ready yet, will retry later"
        return 1
    fi

    local train_log="${LOG_DIR}/${task_id}.log"

    if [ "$task_type" == "act" ]; then
        # Update GPU ID in yaml config
        sed -i "s/CUDA_VISIBLE_DEVICES: \"[0-9]*\"/CUDA_VISIBLE_DEVICES: \"${gpu_id}\"/" "$config_or_args"

        cd "${PROJECT_ROOT}"
        nohup ${PYTHON} "${TRAIN_SCRIPT}" --config "${config_or_args}" &> "${train_log}" &
        GPU_PID[${gpu_id}]=$!
        GPU_TASK[${gpu_id}]="$task_id"
        log "Started ACT [${task_id}] on GPU ${gpu_id}, PID=${GPU_PID[${gpu_id}]}"

    elif [ "$task_type" == "classifier" ]; then
        cd "${PROJECT_ROOT}"
        nohup env CUDA_VISIBLE_DEVICES="${gpu_id}" \
            ${PYTHON} "${VC_ROOT}/scripts/4_train_cluster_classifier.py" \
            --mode "${config_or_args}" &> "${train_log}" &
        GPU_PID[${gpu_id}]=$!
        GPU_TASK[${gpu_id}]="$task_id"
        log "Started classifier [${task_id}] on GPU ${gpu_id}, PID=${GPU_PID[${gpu_id}]}"
    fi
    return 0
}

# ============================================================
# Generate auto configs if not exist
# ============================================================
generate_auto_configs() {
    local auto_config_dir="${VC_ROOT}/config/exp_auto"
    mkdir -p "$auto_config_dir"

    for cluster_id in 0 1; do
        local yaml_path="${auto_config_dir}/cluster${cluster_id}.yaml"
        if [ -f "$yaml_path" ]; then
            continue
        fi
        cat > "$yaml_path" << YAML
# VC-ACT vc_auto: Cluster ${cluster_id}
runtime:
  command: "/venv/mult-act/bin/lerobot-train"
  check_command: true
  cwd: "."
  dry_run: false
  env:
    CUDA_VISIBLE_DEVICES: "0"
    WANDB_API_KEY: "${WANDB_KEY}"

variables:
  dataset_id: "cluster${cluster_id}_v30"
  dataset_root: "${PROJECT_ROOT}/data/vc_auto"
  output_root: "${VC_ROOT}/models/exp_auto"
  run_name: "vc_auto_cluster${cluster_id}"

train_args:
  dataset:
    repo_id: "{dataset_id}"
    root: "{dataset_root}/{dataset_id}"
    video_backend: "torchcodec"

  policy:
    type: "act"
    device: "cuda"
    repo_id: "local/{run_name}"
    push_to_hub: false
    use_amp: true
    n_obs_steps: 1
    chunk_size: 100
    n_action_steps: 100

  steps: 100010
  batch_size: 8
  num_workers: 16
  save_freq: 20000
  log_freq: 200
  wandb:
    mode: "disabled"

  output_dir: "{output_root}/{run_name}"

flags: []
raw_args: []
YAML
        log "Generated config: $yaml_path"
    done
}

# ============================================================
# Main loop
# ============================================================
log "=========================================="
log "VC-ACT Scheduler started"
log "Total tasks: ${#TASKS[@]} (12 ACT + 2 classifier)"
log "GPUs: ${GPUS[*]} (auto-detect free GPUs)"
log "=========================================="

generate_auto_configs

# Deferred tasks (auto_c1 waiting for data)
DEFERRED=()

while true; do
    for gpu_id in "${GPUS[@]}"; do
        # Check if GPU is free
        if [ "${GPU_TASK[${gpu_id}]}" != "idle" ]; then
            if kill -0 "${GPU_PID[${gpu_id}]}" 2>/dev/null; then
                continue  # still running
            fi
            log "GPU ${gpu_id} finished: ${GPU_TASK[${gpu_id}]} (PID ${GPU_PID[${gpu_id}]})"
            GPU_TASK[${gpu_id}]="idle"
            GPU_PID[${gpu_id}]=0
        fi

        # Skip if GPU is occupied by an external process (e.g. MoE-ACT)
        if gpu_has_external_process "$gpu_id"; then
            continue
        fi

        # GPU is idle, try to launch next task

        # First check deferred tasks
        if [ ${#DEFERRED[@]} -gt 0 ]; then
            local_deferred=("${DEFERRED[@]}")
            DEFERRED=()
            for def_task in "${local_deferred[@]}"; do
                if launch_task "$gpu_id" "$def_task"; then
                    break
                else
                    DEFERRED+=("$def_task")
                fi
            done
            [ "${GPU_TASK[${gpu_id}]}" != "idle" ] && continue
        fi

        # Then take from main queue
        if [ ${TASK_IDX} -lt ${#TASKS[@]} ]; then
            task_str="${TASKS[${TASK_IDX}]}"
            TASK_IDX=$((TASK_IDX + 1))

            if ! launch_task "$gpu_id" "$task_str"; then
                DEFERRED+=("$task_str")
            fi
        fi
    done

    # Check if all done
    all_idle=true
    for gpu_id in "${GPUS[@]}"; do
        [ "${GPU_TASK[${gpu_id}]}" != "idle" ] && all_idle=false
    done

    if $all_idle && [ ${TASK_IDX} -ge ${#TASKS[@]} ] && [ ${#DEFERRED[@]} -eq 0 ]; then
        log "=========================================="
        log "All tasks completed!"
        log "=========================================="
        break
    fi

    sleep 30
done
