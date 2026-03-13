#!/usr/bin/env bash
set -e

export WANDB_API_KEY="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"
export CUDA_VISIBLE_DEVICES="0"

PROJECT_ROOT="/workspace/Mult-skill ACT"
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUT_BASE="${PROJECT_ROOT}/models"
TRAIN_SCRIPT="${PROJECT_ROOT}/lerobot/src/lerobot/scripts/lerobot_train.py"

# ---- Shared training hyperparameters ----
BATCH_SIZE=32
STEPS=100010
EVAL_FREQ=20000
LOG_FREQ=200
SAVE_FREQ=20000
SEED=42
CHUNK_SIZE=100
LR=1e-5

MODE=$1

train_act() {
    local DATASET_REPO="$1"
    local DATASET_ROOT="$2"
    local OUTPUT_DIR="$3"
    local RUN_NAME="$4"

    echo "============================================================"
    echo "Training: ${RUN_NAME}"
    echo "  Dataset: ${DATASET_ROOT}"
    echo "  Output:  ${OUTPUT_DIR}"
    echo "============================================================"

    python "${TRAIN_SCRIPT}" \
        --policy.type act \
        --policy.device cuda \
        --policy.chunk_size ${CHUNK_SIZE} \
        --policy.n_action_steps ${CHUNK_SIZE} \
        --policy.dim_model 512 \
        --policy.n_heads 8 \
        --policy.n_encoder_layers 4 \
        --policy.n_decoder_layers 1 \
        --policy.use_vae true \
        --policy.latent_dim 32 \
        --policy.n_vae_encoder_layers 4 \
        --policy.kl_weight 10.0 \
        --policy.optimizer_lr ${LR} \
        --policy.optimizer_weight_decay 1e-4 \
        --policy.vision_backbone resnet18 \
        --policy.pretrained_backbone_weights ResNet18_Weights.IMAGENET1K_V1 \
        --policy.use_amp true \
        --policy.repo_id "local/${RUN_NAME}" \
        --policy.push_to_hub false \
        --dataset.repo_id ${DATASET_REPO} \
        --dataset.root "${DATASET_ROOT}" \
        --dataset.video_backend pyav \
        --batch_size ${BATCH_SIZE} \
        --num_workers 4 \
        --steps ${STEPS} \
        --eval_freq ${EVAL_FREQ} \
        --log_freq ${LOG_FREQ} \
        --save_freq ${SAVE_FREQ} \
        --seed ${SEED} \
        --output_dir "${OUTPUT_DIR}" \
        --job_name "${RUN_NAME}" \
        --wandb.enable true \
        --wandb.project mult-skill-act
}

case ${MODE} in
    all)
        train_act \
            "all_points_v30" \
            "${DATA_DIR}/all_points_v30" \
            "${OUTPUT_BASE}/exp1_all_points" \
            "Exp1_all_points"
        ;;

    all_points)
        echo "Training 10 points sequentially..."
        for i in $(seq 1 10); do
            train_act \
                "point${i}_v30" \
                "${DATA_DIR}/each-point-V30/point${i}_v30" \
                "${OUTPUT_BASE}/exp2_point${i}" \
                "Exp2_point${i}"
        done
        ;;

    act_x)
        echo "============================================================"
        echo "Training: Exp3 ACT-X Conditioned"
        echo "============================================================"

        python "${TRAIN_SCRIPT}" \
            --policy.type act_x \
            --policy.device cuda \
            --policy.chunk_size ${CHUNK_SIZE} \
            --policy.n_action_steps ${CHUNK_SIZE} \
            --policy.dim_model 512 \
            --policy.n_heads 8 \
            --policy.n_encoder_layers 4 \
            --policy.n_decoder_layers 1 \
            --policy.use_vae true \
            --policy.latent_dim 32 \
            --policy.n_vae_encoder_layers 4 \
            --policy.kl_weight 10.0 \
            --policy.optimizer_lr ${LR} \
            --policy.optimizer_weight_decay 1e-4 \
            --policy.vision_backbone resnet18 \
            --policy.pretrained_backbone_weights ResNet18_Weights.IMAGENET1K_V1 \
            --policy.use_amp true \
            --policy.repo_id "local/Exp3_act_x" \
            --policy.push_to_hub false \
            --policy.num_points 10 \
            --dataset.repo_id all_points_v30 \
            --dataset.root "${DATA_DIR}/all_points_v30" \
            --dataset.video_backend pyav \
            --batch_size ${BATCH_SIZE} \
            --num_workers 4 \
            --steps ${STEPS} \
            --eval_freq ${EVAL_FREQ} \
            --log_freq ${LOG_FREQ} \
            --save_freq ${SAVE_FREQ} \
            --seed ${SEED} \
            --output_dir "${OUTPUT_BASE}/exp3_act_x" \
            --job_name "Exp3_act_x" \
            --wandb.enable true \
            --wandb.project mult-skill-act
        ;;

    *)
        echo "Usage:"
        echo "  bash train/train_act.sh all           # Exp1: all 150 eps"
        echo "  bash train/train_act.sh all_points    # Exp2: all 10 points sequentially"
        echo "  bash train/train_act.sh act_x         # Exp3: ACT-X with point condition"
        exit 1
        ;;
esac

echo ""
echo "Training complete!"
