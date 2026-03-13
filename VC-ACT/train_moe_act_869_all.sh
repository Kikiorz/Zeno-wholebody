#!/usr/bin/env bash
# train_moe_act_869_all.sh — 同时训练 3 个 MoE-ACT (不同 chunk_size)
#
# 用法:
#   bash train_moe_act_869_all.sh

set -euo pipefail

PROJECT_ROOT="/workspace/Mult-skill ACT"
VC_ROOT="${PROJECT_ROOT}/VC-ACT"
TRAIN_SCRIPT="${PROJECT_ROOT}/train/run_train.py"
PYTHON="/venv/mult-act/bin/python3"
LOG_DIR="${VC_ROOT}/logs"

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Training 3 MoE-ACT models on hanger-869-v30"
echo "Dataset: 853 episodes, 368K frames"
echo "GPU 1: chunk_size=100"
echo "GPU 2: chunk_size=50"
echo "GPU 3: chunk_size=30"
echo "=========================================="

# Launch GPU 1: chunk_size=100
cd "${PROJECT_ROOT}"
nohup ${PYTHON} "${TRAIN_SCRIPT}" \
  --config "${VC_ROOT}/config/moe_act_869_chunk100.yaml" \
  &> "${LOG_DIR}/moe_act_869_chunk100.log" &
PID1=$!
echo "GPU 1 (chunk=100) started, PID=${PID1}"

# Launch GPU 2: chunk_size=50
nohup ${PYTHON} "${TRAIN_SCRIPT}" \
  --config "${VC_ROOT}/config/moe_act_869_chunk50.yaml" \
  &> "${LOG_DIR}/moe_act_869_chunk50.log" &
PID2=$!
echo "GPU 2 (chunk=50) started, PID=${PID2}"

# Launch GPU 3: chunk_size=30
nohup ${PYTHON} "${TRAIN_SCRIPT}" \
  --config "${VC_ROOT}/config/moe_act_869_chunk30.yaml" \
  &> "${LOG_DIR}/moe_act_869_chunk30.log" &
PID3=$!
echo "GPU 3 (chunk=30) started, PID=${PID3}"

echo ""
echo "All 3 trainings started!"
echo "Monitor logs:"
echo "  tail -f ${LOG_DIR}/moe_act_869_chunk100.log"
echo "  tail -f ${LOG_DIR}/moe_act_869_chunk50.log"
echo "  tail -f ${LOG_DIR}/moe_act_869_chunk30.log"
