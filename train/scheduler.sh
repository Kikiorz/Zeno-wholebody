#!/usr/bin/env bash
# scheduler.sh — 简单调度器：监控 GPU，跑完一个自动安排下一个
# 用法: nohup bash train/scheduler.sh &> train/scheduler.log &

set -uo pipefail

PROJECT_ROOT="/workspace/Mult-skill ACT"
TRAIN_SCRIPT="${PROJECT_ROOT}/train/run_train.py"
CONFIG_DIR="${PROJECT_ROOT}/train/config"
LOG_DIR="${PROJECT_ROOT}/train"
PYTHON="/venv/mult-act/bin/python"

# 当前运行中的任务
declare -A GPU_PID GPU_TASK
GPU_PID[0]=2483296;  GPU_TASK[0]="exp1_all_points"
GPU_PID[1]=1659719;  GPU_TASK[1]="exp2_point1"
GPU_PID[2]=1659847;  GPU_TASK[2]="exp2_point2"
GPU_PID[3]=1659969;  GPU_TASK[3]="exp2_point3"

# 待调度队列
QUEUE=(4 5 6 7 8 9 10)
QUEUE_IDX=0

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

launch_on_gpu() {
    local gpu_id=$1 point_id=$2
    local config="${CONFIG_DIR}/exp2_point${point_id}.yaml"
    local train_log="${LOG_DIR}/exp2_point${point_id}.log"

    sed -i "s/CUDA_VISIBLE_DEVICES: \"[0-9]*\"/CUDA_VISIBLE_DEVICES: \"${gpu_id}\"/" "$config"

    cd "${PROJECT_ROOT}"
    nohup ${PYTHON} "${TRAIN_SCRIPT}" --config "${config}" &> "${train_log}" &
    GPU_PID[${gpu_id}]=$!
    GPU_TASK[${gpu_id}]="exp2_point${point_id}"

    log "Started exp2_point${point_id} on GPU ${gpu_id}, PID=${GPU_PID[${gpu_id}]}"
}

log "Scheduler started. Queue: point${QUEUE[*]}"

while true; do
    for gpu_id in 0 1 2 3; do
        [[ "${GPU_TASK[${gpu_id}]}" == "idle" ]] && continue

        if ! kill -0 "${GPU_PID[${gpu_id}]}" 2>/dev/null; then
            log "GPU ${gpu_id} finished: ${GPU_TASK[${gpu_id}]} (PID ${GPU_PID[${gpu_id}]})"

            if [[ ${QUEUE_IDX} -lt ${#QUEUE[@]} ]]; then
                launch_on_gpu "${gpu_id}" "${QUEUE[${QUEUE_IDX}]}"
                QUEUE_IDX=$((QUEUE_IDX + 1))
            else
                GPU_TASK[${gpu_id}]="idle"
                log "GPU ${gpu_id} idle, queue empty"
            fi
        fi
    done

    # 全部完成则退出
    all_idle=true
    for gpu_id in 0 1 2 3; do
        [[ "${GPU_TASK[${gpu_id}]}" != "idle" ]] && all_idle=false
    done
    $all_idle && [[ ${QUEUE_IDX} -ge ${#QUEUE[@]} ]] && { log "All done!"; break; }

    sleep 30
done
