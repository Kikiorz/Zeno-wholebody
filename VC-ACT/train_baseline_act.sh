#!/usr/bin/env bash
# train_baseline_act.sh — 训练 baseline ACT (100 episodes)
#
# 用法:
#   bash train_baseline_act.sh
#   nohup bash train_baseline_act.sh &> baseline_act.log &

set -euo pipefail

PROJECT_ROOT="/workspace/Mult-skill ACT"
VC_ROOT="${PROJECT_ROOT}/VC-ACT"
TRAIN_SCRIPT="${PROJECT_ROOT}/train/run_train.py"
CONFIG="${VC_ROOT}/config/baseline_act.yaml"
LOG_FILE="${VC_ROOT}/logs/baseline_act.log"
PYTHON="/venv/mult-act/bin/python3"

mkdir -p "${VC_ROOT}/logs"

echo "=========================================="
echo "Training Baseline ACT"
echo "Dataset: 26-03-04-pick_hanger_V30 (100 eps)"
echo "Config: ${CONFIG}"
echo "Log: ${LOG_FILE}"
echo "=========================================="

cd "${PROJECT_ROOT}"
${PYTHON} "${TRAIN_SCRIPT}" --config "${CONFIG}" 2>&1 | tee "${LOG_FILE}"

echo ""
echo "Training completed!"
echo "Model saved to: ${VC_ROOT}/models/baseline_act_100eps"
