#!/usr/bin/env python3
"""
Pseudo-Label Generation for Stage A→B Switch Boundary Detection.

For each full-task trajectory (stage3_lerobot, 20 episodes), finds the optimal
timestep τ* to switch from stageA expert to stageB expert by minimizing:
    C(τ) = Σ_{t≤τ} e_A(t) + Σ_{t>τ} e_B(t)

Outputs per-episode pkl files, diagnostic plots, and a summary JSON.
"""

import argparse
import gc
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file as load_safetensors

# Add lerobot to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lerobot" / "src"))

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.video_utils import decode_video_frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT / "data" / "Exp_stageAB"
STAGE3_DIR = DATA_ROOT / "stage3_lerobot"
MODEL_ROOT = PROJECT / "model"

STAGE_A_CKPT = MODEL_ROOT / "stage1_act" / "checkpoints" / "last" / "pretrained_model"
STAGE_B_CKPT = MODEL_ROOT / "stage2_act" / "checkpoints" / "last" / "pretrained_model"

OUTPUT_ROOT = DATA_ROOT / "pseudo_labels"

# ImageNet normalization (same for both experts)
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)

CAMERAS = ["realsense_top", "realsense_left", "realsense_right"]
CHUNK_SIZE = 100
ACTION_DIM = 17
FPS = 30


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_norm_stats(ckpt_dir: Path):
    """Load action/state mean+std from safetensors files in a checkpoint dir."""
    post_file = ckpt_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    pre_file = ckpt_dir / "policy_preprocessor_step_3_normalizer_processor.safetensors"

    post = load_safetensors(str(post_file))
    pre = load_safetensors(str(pre_file))

    # Clamp std to 1e-6 to avoid division-by-zero (some dims like base_omega have std=0)
    EPS = 1e-6
    return {
        "action_mean": post["action.mean"],
        "action_std": post["action.std"].clamp(min=EPS),
        "state_mean": pre["observation.state.mean"],
        "state_std": pre["observation.state.std"].clamp(min=EPS),
    }


def load_expert(ckpt_dir: Path, device: str = "cuda"):
    """Load an ACT policy from a pretrained checkpoint directory."""
    policy = ACTPolicy.from_pretrained(str(ckpt_dir))
    policy.to(device)
    policy.eval()
    return policy


def load_episode_metadata():
    """Load episode metadata from stage3 parquet."""
    ep_pq = STAGE3_DIR / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(str(ep_pq))
    df = table.to_pandas()
    return df


def load_episode_data(ep_idx: int, all_data, ep_meta):
    """Extract state and action arrays for a single episode from the full data table."""
    row = ep_meta.loc[ep_meta["episode_index"] == ep_idx].iloc[0]
    from_idx = int(row["dataset_from_index"])
    to_idx = int(row["dataset_to_index"])

    states = np.array(all_data["observation.state"][from_idx:to_idx].to_pylist(), dtype=np.float32)
    actions = np.array(all_data["action"][from_idx:to_idx].to_pylist(), dtype=np.float32)
    timestamps = np.array(all_data["timestamp"][from_idx:to_idx].to_pylist(), dtype=np.float64)

    return states, actions, timestamps


def decode_episode_frames(ep_idx: int, ep_meta, timestamps: np.ndarray):
    """Decode video frames for all 3 cameras for an episode.

    Returns dict: camera_name -> (T, 3, 224, 224) float32 tensor in [0,1].
    """
    row = ep_meta.loc[ep_meta["episode_index"] == ep_idx].iloc[0]
    frames = {}

    for cam in CAMERAS:
        chunk_idx = int(row[f"videos/observation.images.{cam}/chunk_index"])
        file_idx = int(row[f"videos/observation.images.{cam}/file_index"])
        from_ts = float(row[f"videos/observation.images.{cam}/from_timestamp"])

        video_path = STAGE3_DIR / "videos" / f"observation.images.{cam}" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"

        # Timestamps within the video file = episode-local timestamps + from_ts offset
        video_timestamps = (timestamps + from_ts).tolist()

        cam_frames = decode_video_frames(
            video_path=video_path,
            timestamps=video_timestamps,
            tolerance_s=1.0 / FPS + 0.01,
            backend="pyav",
        )
        # cam_frames: (T, C, H, W) float32 in [0,1]
        frames[cam] = cam_frames

    return frames


def normalize_images(frames_dict: dict, device: str = "cuda"):
    """Apply ImageNet normalization to frames. Returns dict of (T, 3, 224, 224) tensors on device."""
    img_mean = IMAGE_MEAN.to(device)
    img_std = IMAGE_STD.to(device)
    result = {}
    for cam, frames in frames_dict.items():
        f = frames.to(device)
        f = (f - img_mean) / img_std
        result[cam] = f
    return result


def normalize_state(states: np.ndarray, stats: dict, device: str = "cuda"):
    """Normalize states using expert's own stats. Returns (T, 17) tensor on device."""
    s_mean = stats["state_mean"].to(device)
    s_std = stats["state_std"].to(device)
    s = torch.from_numpy(states).to(device)
    return (s - s_mean) / s_std


def denormalize_actions(actions_norm: torch.Tensor, stats: dict):
    """Denormalize predicted actions to physical space. Input: (B, chunk, 17) on any device."""
    a_mean = stats["action_mean"].to(actions_norm.device)
    a_std = stats["action_std"].to(actions_norm.device)
    return actions_norm * a_std + a_mean


@torch.no_grad()
def run_expert_batched(policy, norm_images: dict, norm_states: torch.Tensor,
                       stats: dict, batch_size: int = 32):
    """Run expert on all timesteps in mini-batches.

    Returns: (T, chunk_size, action_dim) numpy array in physical space.
    """
    T = norm_states.shape[0]
    device = norm_states.device
    all_actions = []

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        B = end - start

        batch = {
            "observation.state": norm_states[start:end],  # (B, 17)
        }
        for cam in CAMERAS:
            batch[f"observation.images.{cam}"] = norm_images[cam][start:end]  # (B, 3, 224, 224)

        # predict_action_chunk returns (B, chunk_size, action_dim) normalized
        actions_norm = policy.predict_action_chunk(batch)  # (B, 100, 17)

        # Denormalize to physical space
        actions_phys = denormalize_actions(actions_norm, stats)
        all_actions.append(actions_phys.cpu().numpy())

    return np.concatenate(all_actions, axis=0)  # (T, 100, 17)


def build_gt_action_chunks(actions: np.ndarray):
    """Build ground-truth action chunks from the trajectory.

    For timestep t, the GT chunk is actions[t:t+chunk_size].
    At the end of the trajectory, pad by repeating the last action.

    Returns: (T, chunk_size, action_dim) numpy array.
    """
    T = actions.shape[0]
    chunks = np.zeros((T, CHUNK_SIZE, ACTION_DIM), dtype=np.float32)

    for t in range(T):
        remaining = min(CHUNK_SIZE, T - t)
        chunks[t, :remaining] = actions[t:t + remaining]
        if remaining < CHUNK_SIZE:
            chunks[t, remaining:] = actions[-1]  # pad with last action

    return chunks


def compute_chunk_errors(pred_chunks: np.ndarray, gt_chunks: np.ndarray):
    """Compute per-timestep L2 chunk error.

    e(t) = mean over h of ||pred[t,h,:] - gt[t,h,:]||_2

    Returns: (T,) numpy array.
    """
    # (T, chunk_size, action_dim) -> per-step L2 norms
    diff = pred_chunks - gt_chunks  # (T, 100, 17)
    l2_per_step = np.linalg.norm(diff, axis=2)  # (T, 100)
    return l2_per_step.mean(axis=1)  # (T,)


def find_optimal_switch(e_A: np.ndarray, e_B: np.ndarray):
    """Find τ* = argmin_τ C(τ) where C(τ) = Σ_{t≤τ} e_A(t) + Σ_{t>τ} e_B(t).

    Uses cumsum for O(T) computation.
    Returns: tau_star, cost_curve (T,)
    """
    T = len(e_A)
    cum_A = np.cumsum(e_A)       # cum_A[τ] = Σ_{t=0..τ} e_A(t)
    total_B = np.sum(e_B)
    cum_B = np.cumsum(e_B)       # cum_B[τ] = Σ_{t=0..τ} e_B(t)

    # C(τ) = cum_A[τ] + (total_B - cum_B[τ])
    cost = cum_A + (total_B - cum_B)

    tau_star = int(np.argmin(cost))
    return tau_star, cost


def generate_labels(T: int, tau_star: int, e_A: np.ndarray, e_B: np.ndarray,
                    gamma: float = 0.1, ambiguity_window: int = 50):
    """Generate hard labels, soft labels, and ambiguity mask.

    Hard labels: 0 for t <= tau_star, 1 for t > tau_star
    Soft labels: softmax(-e_k / gamma) normalized between A and B per timestep
    Ambiguity mask: 1 for timesteps within ambiguity_window of tau_star
    """
    hard = np.zeros(T, dtype=np.int32)
    hard[tau_star + 1:] = 1

    # Soft labels: P(B|t) = exp(-e_B(t)/γ) / (exp(-e_A(t)/γ) + exp(-e_B(t)/γ))
    logit_A = -e_A / gamma
    logit_B = -e_B / gamma
    # Numerically stable softmax
    max_logit = np.maximum(logit_A, logit_B)
    exp_A = np.exp(logit_A - max_logit)
    exp_B = np.exp(logit_B - max_logit)
    soft_B = exp_B / (exp_A + exp_B)  # P(use expert B)

    # Ambiguity mask
    ambiguity = np.zeros(T, dtype=np.int32)
    lo = max(0, tau_star - ambiguity_window)
    hi = min(T, tau_star + ambiguity_window + 1)
    ambiguity[lo:hi] = 1

    return hard, soft_B, ambiguity


def plot_episode(ep_idx: int, e_A: np.ndarray, e_B: np.ndarray,
                 cost: np.ndarray, tau_star: int, soft_B: np.ndarray,
                 save_path: Path):
    """Generate 3-panel diagnostic plot."""
    T = len(e_A)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: Expert errors
    ax = axes[0]
    ax.plot(e_A, label="e_A (stage1)", alpha=0.8, linewidth=0.5)
    ax.plot(e_B, label="e_B (stage2)", alpha=0.8, linewidth=0.5)
    ax.axvline(tau_star, color="red", linestyle="--", label=f"τ*={tau_star}")
    ax.set_ylabel("L2 chunk error")
    ax.set_title(f"Episode {ep_idx} — Expert Errors (τ*={tau_star}, T={T})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Cost curve
    ax = axes[1]
    ax.plot(cost, color="purple", linewidth=0.8)
    ax.axvline(tau_star, color="red", linestyle="--")
    ax.scatter([tau_star], [cost[tau_star]], color="red", zorder=5, s=40)
    ax.set_ylabel("C(τ)")
    ax.set_title("Cost Curve")
    ax.grid(True, alpha=0.3)

    # Panel 3: Soft labels
    ax = axes[2]
    ax.fill_between(range(T), soft_B, alpha=0.5, color="orange", label="P(stage2)")
    ax.fill_between(range(T), 1 - soft_B, alpha=0.5, color="blue", label="P(stage1)")
    ax.axvline(tau_star, color="red", linestyle="--")
    ax.set_ylabel("Soft label")
    ax.set_xlabel("Timestep")
    ax.set_title("Soft Labels (P(expert))")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────────────────

def process_episode(ep_idx: int, policy_A, policy_B, stats_A, stats_B,
                    all_data, ep_meta, args):
    """Process a single episode: run experts, compute errors, find switch point."""
    logger.info(f"=== Episode {ep_idx} ===")
    t0 = time.time()

    # 1. Load episode data
    states, actions, timestamps = load_episode_data(ep_idx, all_data, ep_meta)
    T = len(states)
    logger.info(f"  T={T} frames, timestamps [{timestamps[0]:.3f}, {timestamps[-1]:.3f}]")

    # 2. Decode video frames
    logger.info("  Decoding video frames...")
    t_vid = time.time()
    frames = decode_episode_frames(ep_idx, ep_meta, timestamps)
    logger.info(f"  Video decode: {time.time() - t_vid:.1f}s")

    # 3. Normalize images
    norm_images = normalize_images(frames, device=args.device)
    del frames
    gc.collect()

    # 4. Normalize states for each expert
    norm_states_A = normalize_state(states, stats_A, device=args.device)
    norm_states_B = normalize_state(states, stats_B, device=args.device)

    # 5. Run experts
    logger.info("  Running expert A...")
    t_inf = time.time()
    pred_A = run_expert_batched(policy_A, norm_images, norm_states_A, stats_A, batch_size=args.batch_size)
    logger.info(f"  Expert A: {time.time() - t_inf:.1f}s")

    logger.info("  Running expert B...")
    t_inf = time.time()
    pred_B = run_expert_batched(policy_B, norm_images, norm_states_B, stats_B, batch_size=args.batch_size)
    logger.info(f"  Expert B: {time.time() - t_inf:.1f}s")

    # Free GPU memory
    del norm_images, norm_states_A, norm_states_B
    torch.cuda.empty_cache()
    gc.collect()

    # 6. Build GT action chunks (already in physical space)
    gt_chunks = build_gt_action_chunks(actions)

    # 7. Compute errors
    e_A = compute_chunk_errors(pred_A, gt_chunks)
    e_B = compute_chunk_errors(pred_B, gt_chunks)

    # 8. Find optimal switch
    tau_star, cost = find_optimal_switch(e_A, e_B)
    logger.info(f"  τ*={tau_star} ({tau_star/T*100:.1f}% of trajectory)")

    # 9. Generate labels
    hard, soft_B, ambiguity = generate_labels(T, tau_star, e_A, e_B,
                                               gamma=args.gamma,
                                               ambiguity_window=args.ambiguity_window)

    # 10. Save episode results
    ep_dir = OUTPUT_ROOT / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)
    ep_result = {
        "episode_index": ep_idx,
        "T": T,
        "tau_star": tau_star,
        "tau_star_frac": tau_star / T,
        "cost_at_tau": float(cost[tau_star]),
        "e_A": e_A,
        "e_B": e_B,
        "cost_curve": cost,
        "hard_labels": hard,
        "soft_labels": soft_B,
        "ambiguity_mask": ambiguity,
        "mean_eA_before_tau": float(e_A[:tau_star + 1].mean()) if tau_star > 0 else float(e_A[0]),
        "mean_eB_after_tau": float(e_B[tau_star + 1:].mean()) if tau_star < T - 1 else float(e_B[-1]),
    }
    pkl_path = ep_dir / f"episode_{ep_idx:03d}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(ep_result, f)

    # 11. Plot
    plot_dir = OUTPUT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_episode(ep_idx, e_A, e_B, cost, tau_star, soft_B,
                 plot_dir / f"episode_{ep_idx:03d}.png")

    elapsed = time.time() - t0
    logger.info(f"  Done in {elapsed:.1f}s")

    return {
        "episode_index": ep_idx,
        "T": T,
        "tau_star": tau_star,
        "tau_star_frac": round(tau_star / T, 4),
        "cost_at_tau": round(float(cost[tau_star]), 4),
        "mean_eA_before_tau": round(ep_result["mean_eA_before_tau"], 4),
        "mean_eB_after_tau": round(ep_result["mean_eB_after_tau"], 4),
        "elapsed_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-labels for A→B switch boundary")
    parser.add_argument("--episodes", type=str, default=None,
                        help="Comma-separated episode indices or range (e.g. '0' or '0,1,5' or '0-19'). Default: all 20.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="Temperature for soft labels")
    parser.add_argument("--ambiguity-window", type=int, default=50,
                        help="Half-width of ambiguity mask around τ*")
    parser.add_argument("--stage-a-ckpt", type=str, default=None,
                        help="Override stageA checkpoint path")
    parser.add_argument("--stage-b-ckpt", type=str, default=None,
                        help="Override stageB checkpoint path")
    args = parser.parse_args()

    ckpt_A = Path(args.stage_a_ckpt) if args.stage_a_ckpt else STAGE_A_CKPT
    ckpt_B = Path(args.stage_b_ckpt) if args.stage_b_ckpt else STAGE_B_CKPT

    # Parse episode list
    if args.episodes is None:
        episode_list = list(range(20))
    elif "-" in args.episodes and "," not in args.episodes:
        lo, hi = args.episodes.split("-")
        episode_list = list(range(int(lo), int(hi) + 1))
    else:
        episode_list = [int(x) for x in args.episodes.split(",")]

    logger.info(f"Episodes to process: {episode_list}")
    logger.info(f"Stage A checkpoint: {ckpt_A}")
    logger.info(f"Stage B checkpoint: {ckpt_B}")

    # Load models
    logger.info("Loading expert A (stage1)...")
    policy_A = load_expert(ckpt_A, device=args.device)
    stats_A = load_norm_stats(ckpt_A)

    logger.info("Loading expert B (stage2)...")
    policy_B = load_expert(ckpt_B, device=args.device)
    stats_B = load_norm_stats(ckpt_B)

    # Load data
    logger.info("Loading stage3 data...")
    data_pq = STAGE3_DIR / "data" / "chunk-000" / "file-000.parquet"
    all_data = pq.read_table(str(data_pq))
    ep_meta = load_episode_metadata()

    # Process episodes
    summaries = []
    for ep_idx in episode_list:
        summary = process_episode(ep_idx, policy_A, policy_B, stats_A, stats_B,
                                  all_data, ep_meta, args)
        summaries.append(summary)

    # Write summary
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    tau_stars = [s["tau_star"] for s in summaries]
    tau_fracs = [s["tau_star_frac"] for s in summaries]

    summary_json = {
        "config": {
            "stage_a_ckpt": str(ckpt_A),
            "stage_b_ckpt": str(ckpt_B),
            "gamma": args.gamma,
            "ambiguity_window": args.ambiguity_window,
            "batch_size": args.batch_size,
            "chunk_size": CHUNK_SIZE,
        },
        "global_stats": {
            "n_episodes": len(summaries),
            "tau_star_mean": round(float(np.mean(tau_stars)), 1),
            "tau_star_std": round(float(np.std(tau_stars)), 1),
            "tau_star_frac_mean": round(float(np.mean(tau_fracs)), 4),
            "tau_star_frac_std": round(float(np.std(tau_fracs)), 4),
            "tau_star_min": int(np.min(tau_stars)),
            "tau_star_max": int(np.max(tau_stars)),
        },
        "episodes": summaries,
    }

    summary_path = OUTPUT_ROOT / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_json, f, indent=2)

    logger.info(f"\nResults saved to {OUTPUT_ROOT}")
    logger.info(f"τ* mean={np.mean(tau_stars):.1f} ± {np.std(tau_stars):.1f} "
                f"(frac: {np.mean(tau_fracs):.3f} ± {np.std(tau_fracs):.3f})")


if __name__ == "__main__":
    main()
