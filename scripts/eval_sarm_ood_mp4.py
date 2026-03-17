#!/usr/bin/env python3
"""
Evaluate SARM on out-of-distribution stage1 data and generate MP4 visualization.

Uses the same inference pipeline as compute_rabc_weights.py (LeRobotDataset + preprocessor)
to ensure correct bidirectional sampling and CLIP encoding.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "lerobot" / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.policies.sarm.compute_rabc_weights import load_sarm_resources, to_numpy_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sarm-ckpt", type=str,
                        default=str(PROJECT_ROOT / "model/sarm/train/checkpoints/last/pretrained_model"))
    parser.add_argument("--dataset-repo-id", type=str, default="stage1_lerobot")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "model/sarm/eval/sarm_ood_stage1_ep0.mp4"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--head-mode", type=str, default="dense")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set dataset home to find stage1_lerobot
    os.environ["HF_LEROBOT_HOME"] = str(PROJECT_ROOT / "data" / "Exp_stageAB")

    # Load using the same pipeline as compute_rabc_weights.py
    print("Loading SARM resources...")
    dataset, reward_model, preprocess = load_sarm_resources(
        dataset_repo_id=args.dataset_repo_id,
        reward_model_path=args.sarm_ckpt,
        device=args.device,
    )

    # Set preprocessor to eval mode
    if hasattr(preprocess, "eval"):
        preprocess.eval()
    for step in preprocess.steps:
        if hasattr(step, "eval"):
            step.eval()

    image_key = reward_model.config.image_key
    state_key = reward_model.config.state_key
    device = reward_model.device
    target_idx = reward_model.config.n_obs_steps // 2  # center frame

    # Get episode info
    ep = dataset.meta.episodes[args.episode]
    ep_start = ep["dataset_from_index"]
    ep_end = ep["dataset_to_index"]
    num_frames = ep_end - ep_start
    task = dataset[ep_start].get("task", "perform the task")
    print(f"Episode {args.episode}: {num_frames} frames, task='{task}'")

    # Determine stage info
    head_mode = args.head_mode
    if head_mode == "dense" and reward_model.config.uses_dual_heads:
        num_stages = reward_model.config.num_dense_stages
        stage_names = reward_model.config.dense_subtask_names
    else:
        head_mode = "sparse"
        num_stages = reward_model.config.num_sparse_stages
        stage_names = reward_model.config.sparse_subtask_names
    print(f"Head mode: {head_mode}, stages: {stage_names}")

    # Run inference on every frame
    all_progress = np.zeros(num_frames)
    all_stages = np.zeros((num_frames, num_stages))
    all_frames = []

    print("Running SARM inference...")
    for frame_idx in tqdm(range(ep_start, ep_end), desc="Inference"):
        local_idx = frame_idx - ep_start
        sample = dataset[frame_idx]

        # Get display frame
        img = to_numpy_image(sample[image_key])
        all_frames.append(img)

        # Prepare batch (same as compute_rabc_weights.py)
        batch = {
            image_key: sample[image_key],
            "task": task,
            "index": frame_idx,
            "episode_index": args.episode,
        }
        if state_key in sample:
            batch[state_key] = sample[state_key]

        with torch.no_grad():
            processed = preprocess(batch)
            video_features = processed["video_features"].to(device)
            text_features = processed["text_features"].to(device)
            state_features = processed.get("state_features")
            if state_features is not None:
                state_features = state_features.to(device)
            lengths = processed.get("lengths")

            reward, stage_probs = reward_model.calculate_rewards(
                text_embeddings=text_features,
                video_embeddings=video_features,
                state_features=state_features,
                lengths=lengths,
                return_all_frames=True,
                return_stages=True,
                head_mode=head_mode,
            )

            if isinstance(reward, torch.Tensor):
                reward = reward.cpu().numpy()
                stage_probs = stage_probs.cpu().numpy()

            if reward.ndim == 2:
                all_progress[local_idx] = reward[0, target_idx]
                all_stages[local_idx] = stage_probs[0, target_idx, :]
            else:
                all_progress[local_idx] = reward[target_idx]
                all_stages[local_idx] = stage_probs[target_idx, :]

            del processed, video_features, text_features
            if state_features is not None:
                del state_features

        if local_idx % 100 == 0:
            torch.cuda.empty_cache()

    print(f"Inference done. Writing MP4...")

    # Write MP4 with overlaid predictions
    H, W = all_frames[0].shape[:2]
    canvas_h = H + 80  # extra space for overlay
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, 30.0, (W, canvas_h))

    for i in range(num_frames):
        frame_rgb = all_frames[i]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Create canvas with black bottom bar
        canvas = np.zeros((canvas_h, W, 3), dtype=np.uint8)
        canvas[:H, :W] = frame_bgr

        p_hang = all_stages[i, 0] if num_stages > 0 else 0
        p_move = all_stages[i, 1] if num_stages > 1 else 0
        progress = all_progress[i]

        # Stage probability bar
        bar_y = H + 5
        bar_w = W - 10
        bar_h = 15
        hang_w = int(bar_w * p_hang)
        cv2.rectangle(canvas, (5, bar_y), (5 + hang_w, bar_y + bar_h), (0, 200, 0), -1)
        cv2.rectangle(canvas, (5 + hang_w, bar_y), (5 + bar_w, bar_y + bar_h), (0, 0, 200), -1)
        cv2.putText(canvas, f"hang:{p_hang:.2f}", (5, bar_y - 3),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1)
        cv2.putText(canvas, f"move:{p_move:.2f}", (W - 80, bar_y - 3),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)

        # Progress bar
        prog_y = bar_y + bar_h + 10
        prog_w = int(bar_w * min(max(progress, 0), 1.0))
        cv2.rectangle(canvas, (5, prog_y), (5 + bar_w, prog_y + 15), (50, 50, 50), -1)
        cv2.rectangle(canvas, (5, prog_y), (5 + prog_w, prog_y + 15), (255, 200, 0), -1)
        cv2.putText(canvas, f"progress: {progress:.3f}", (5, prog_y + 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

        # Stage label
        stage_label = stage_names[np.argmax(all_stages[i])]
        color = (0, 255, 0) if np.argmax(all_stages[i]) == 0 else (0, 0, 255)
        cv2.putText(canvas, stage_label.upper(), (5, 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Frame counter
        cv2.putText(canvas, f"f:{i}/{num_frames}", (W - 90, 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        writer.write(canvas)

    writer.release()
    print(f"Done! Output: {output_path}")
    print(f"\nSummary:")
    print(f"  Mean P({stage_names[0]}): {np.mean(all_stages[:, 0]):.4f}")
    if num_stages > 1:
        print(f"  Mean P({stage_names[1]}): {np.mean(all_stages[:, 1]):.4f}")
    print(f"  Progress range: {all_progress.min():.4f} - {all_progress.max():.4f}")
    print(f"  Final progress: {all_progress[-1]:.4f}")


if __name__ == "__main__":
    main()
