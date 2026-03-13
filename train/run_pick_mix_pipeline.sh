#!/usr/bin/env bash
# run_pick_mix_pipeline.sh — 自动化数据转换 + 双模型训练流程
# 用法: nohup bash train/run_pick_mix_pipeline.sh &> train/pipeline.log &

set -uo pipefail

PROJECT_ROOT="/workspace/Mult-skill ACT"
TRAIN_SCRIPT="${PROJECT_ROOT}/train/run_train.py"
CONFIG_DIR="${PROJECT_ROOT}/train/config"
LOG_DIR="${PROJECT_ROOT}/train"
PYTHON="/venv/mult-act/bin/python"
DATA_DIR="${PROJECT_ROOT}/data/26-03-01-pick-mix"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# GPU 检测函数
get_free_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | \
    awk -F', ' '$2 < 1000 {print $1}'
}

wait_for_free_gpu() {
  while true; do
    local free=$(get_free_gpus | head -1)
    if [[ -n "$free" ]]; then
      echo "$free"
      return
    fi
    log "No free GPU, waiting 30s..."
    sleep 30
  done
}

# 等待数据转换完成
log "=== 步骤 1: 等待数据转换完成 ==="
while true; do
  # 检查 bag2lerobot 进程是否还在运行
  if pgrep -f "bag2lerobot_v30.py.*26-03-01-pick-mix" > /dev/null; then
    log "数据转换进行中，等待 60s..."
    sleep 60
  else
    # 进程结束，检查数据集是否完整
    if [[ -f "${DATA_DIR}/meta/info.json" ]] && [[ -d "${DATA_DIR}/data" ]] && [[ -d "${DATA_DIR}/videos" ]]; then
      log "数据转换完成"
      break
    else
      log "错误：数据转换进程已结束，但数据集不完整"
      exit 1
    fi
  fi
done

# 验证数据集
if [[ ! -d "${DATA_DIR}/data" ]] || [[ ! -d "${DATA_DIR}/videos" ]]; then
  log "错误：数据集目录不完整"
  exit 1
fi

log "数据集验证通过"

# 启动 ACT 训练
log "=== 步骤 2: 启动 ACT 训练 ==="
GPU_ACT=$(wait_for_free_gpu)
log "找到空闲 GPU ${GPU_ACT}，启动 ACT 训练"

# 更新配置文件中的 GPU
sed -i "s/CUDA_VISIBLE_DEVICES: \"[0-9]*\"/CUDA_VISIBLE_DEVICES: \"${GPU_ACT}\"/" "${CONFIG_DIR}/pick_mix_act.yaml"

cd "${PROJECT_ROOT}"
nohup ${PYTHON} "${TRAIN_SCRIPT}" --config "${CONFIG_DIR}/pick_mix_act.yaml" &> "${LOG_DIR}/pick_mix_act.log" &
PID_ACT=$!
log "ACT 训练已启动，PID=${PID_ACT}, GPU=${GPU_ACT}"

# 启动 MoE-ACT 训练
log "=== 步骤 3: 启动 MoE-ACT 训练 ==="
GPU_MOE=$(wait_for_free_gpu)
log "找到空闲 GPU ${GPU_MOE}，启动 MoE-ACT 训练"

# 更新配置文件中的 GPU
sed -i "s/CUDA_VISIBLE_DEVICES: \"[0-9]*\"/CUDA_VISIBLE_DEVICES: \"${GPU_MOE}\"/" "${CONFIG_DIR}/pick_mix_moe_act.yaml"

nohup ${PYTHON} "${TRAIN_SCRIPT}" --config "${CONFIG_DIR}/pick_mix_moe_act.yaml" &> "${LOG_DIR}/pick_mix_moe_act.log" &
PID_MOE=$!
log "MoE-ACT 训练已启动，PID=${PID_MOE}, GPU=${GPU_MOE}"

# 监控训练进程
log "=== 步骤 4: 监控训练进程 ==="
log "ACT 训练: PID=${PID_ACT}, GPU=${GPU_ACT}, 日志: ${LOG_DIR}/pick_mix_act.log"
log "MoE-ACT 训练: PID=${PID_MOE}, GPU=${GPU_MOE}, 日志: ${LOG_DIR}/pick_mix_moe_act.log"

# 等待两个训练完成
wait $PID_ACT
ACT_EXIT=$?
log "ACT 训练完成，退出码: ${ACT_EXIT}"

wait $PID_MOE
MOE_EXIT=$?
log "MoE-ACT 训练完成，退出码: ${MOE_EXIT}"

# 总结
log "=== 所有任务完成 ==="
log "ACT 模型输出: ${PROJECT_ROOT}/models/pick_mix_act/"
log "MoE-ACT 模型输出: ${PROJECT_ROOT}/models/pick_mix_moe_act/"

if [[ $ACT_EXIT -eq 0 ]] && [[ $MOE_EXIT -eq 0 ]]; then
  log "✓ 所有训练成功完成"
  exit 0
else
  log "✗ 部分训练失败，请检查日志"
  exit 1
fi
