#!/usr/bin/env python3
"""
Convert pseudo-labels (tau_star, T) into SARM-compatible annotations.

Writes:
  1. meta/temporal_proportions_dense.json  — mean temporal proportions for 2 dense stages
  2. meta/temporal_proportions_sparse.json — single "task" stage
  3. Updates episodes parquet with dense_subtask_* columns

Usage:
    /venv/main/bin/python scripts/prepare_sarm_annotations.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PSEUDO_LABELS_DIR = PROJECT_ROOT / "data" / "Exp_stageAB" / "pseudo_labels" / "episodes"
DATASET_DIR = PROJECT_ROOT / "data" / "Exp_stageAB" / "stage3_lerobot"
META_DIR = DATASET_DIR / "meta"
EPISODES_PARQUET = META_DIR / "episodes" / "chunk-000" / "file-000.parquet"

STAGE_NAMES = ["hang_clothes", "move_hanger"]
FPS = 30.0
NUM_EPISODES = 20


def load_pseudo_labels():
    """Load tau_star and T from all 20 episode pickle files."""
    episodes = []
    for ep_idx in range(NUM_EPISODES):
        pkl_path = PSEUDO_LABELS_DIR / f"episode_{ep_idx:03d}.pkl"
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        episodes.append({
            "episode_index": data["episode_index"],
            "tau_star": int(data["tau_star"]),
            "T": int(data["T"]),
        })
    return episodes


def compute_temporal_proportions(episodes):
    """Compute mean temporal proportions: hang_clothes fraction = mean(tau_star / T)."""
    fracs = [ep["tau_star"] / ep["T"] for ep in episodes]
    hang_frac = float(np.mean(fracs))
    move_frac = 1.0 - hang_frac
    return {STAGE_NAMES[0]: round(hang_frac, 4), STAGE_NAMES[1]: round(move_frac, 4)}


def write_temporal_proportions(proportions):
    """Write both dense and sparse temporal proportions JSON files."""
    # Dense: 2 stages with computed proportions
    dense_path = META_DIR / "temporal_proportions_dense.json"
    with open(dense_path, "w") as f:
        json.dump(proportions, f, indent=2)
    print(f"Wrote {dense_path}")
    print(f"  Dense proportions: {proportions}")

    # Sparse: single "task" stage
    sparse_path = META_DIR / "temporal_proportions_sparse.json"
    sparse = {"task": 1.0}
    with open(sparse_path, "w") as f:
        json.dump(sparse, f, indent=2)
    print(f"Wrote {sparse_path}")


def update_episodes_parquet(episodes):
    """Add dense_subtask_* columns to the episodes parquet file."""
    df = pd.read_parquet(EPISODES_PARQUET)
    print(f"Read episodes parquet: {df.shape[0]} rows, {df.shape[1]} columns")

    # Build annotation columns for each episode
    dense_subtask_names = []
    dense_subtask_start_frames = []
    dense_subtask_end_frames = []
    dense_subtask_start_times = []
    dense_subtask_end_times = []

    for _, row in df.iterrows():
        ep_idx = int(row["episode_index"])
        ep_data = episodes[ep_idx]
        tau_star = ep_data["tau_star"]
        T = ep_data["T"]

        # Stage boundaries (inclusive on both sides for find_stage_and_tau):
        #   hang_clothes [0, tau_star-1], move_hanger [tau_star, T-1]
        dense_subtask_names.append(STAGE_NAMES)
        dense_subtask_start_frames.append([0, tau_star])
        dense_subtask_end_frames.append([tau_star - 1, T - 1])
        dense_subtask_start_times.append([0.0, tau_star / FPS])
        dense_subtask_end_times.append([(tau_star - 1) / FPS, (T - 1) / FPS])

    df["dense_subtask_names"] = dense_subtask_names
    df["dense_subtask_start_frames"] = dense_subtask_start_frames
    df["dense_subtask_end_frames"] = dense_subtask_end_frames
    df["dense_subtask_start_times"] = dense_subtask_start_times
    df["dense_subtask_end_times"] = dense_subtask_end_times

    # Write back as parquet preserving existing data
    df.to_parquet(EPISODES_PARQUET, index=False)
    print(f"Updated {EPISODES_PARQUET} with {len(df.columns)} columns")
    print(f"  New columns: dense_subtask_names, dense_subtask_start_frames, "
          f"dense_subtask_end_frames, dense_subtask_start_times, dense_subtask_end_times")


def verify(episodes):
    """Quick verification that annotations load correctly."""
    from lerobot.policies.sarm.sarm_utils import find_stage_and_tau

    df = pd.read_parquet(EPISODES_PARQUET)
    proportions = json.load(open(META_DIR / "temporal_proportions_dense.json"))
    global_names = list(proportions.keys())

    for ep_idx in [0, 10, 19]:
        row = df.loc[ep_idx]
        tau_star = episodes[ep_idx]["tau_star"]
        T = episodes[ep_idx]["T"]

        names = row["dense_subtask_names"]
        starts = row["dense_subtask_start_frames"]
        ends = row["dense_subtask_end_frames"]

        # Frame 0: should be stage 0 (hang_clothes), tau ≈ 0
        stage, tau = find_stage_and_tau(0, T, names, starts, ends, global_names, proportions)
        print(f"  Episode {ep_idx}: frame=0 → stage={stage} ({global_names[stage]}), tau={tau:.4f}")
        assert stage == 0 and tau < 0.01, f"Expected stage=0, tau≈0 at frame 0, got {stage}, {tau}"

        # Frame tau_star: should be stage 1 (move_hanger), tau ≈ 0
        stage, tau = find_stage_and_tau(tau_star, T, names, starts, ends, global_names, proportions)
        print(f"  Episode {ep_idx}: frame={tau_star} (tau*) → stage={stage} ({global_names[stage]}), tau={tau:.4f}")
        assert stage == 1 and tau < 0.01, f"Expected stage=1, tau≈0 at tau*, got {stage}, {tau}"

        # Frame T-1: should be stage 1 (move_hanger), tau close to 1
        stage, tau = find_stage_and_tau(T - 1, T, names, starts, ends, global_names, proportions)
        print(f"  Episode {ep_idx}: frame={T-1} (end) → stage={stage} ({global_names[stage]}), tau={tau:.4f}")
        assert stage == 1 and tau > 0.9, f"Expected stage=1, tau≈1 at end, got {stage}, {tau}"

    print("\nVerification passed!")


def main():
    print("=" * 60)
    print("Preparing SARM annotations from pseudo-labels")
    print("=" * 60)

    # Load pseudo-labels
    episodes = load_pseudo_labels()
    print(f"\nLoaded {len(episodes)} episodes:")
    for ep in episodes[:3]:
        print(f"  Episode {ep['episode_index']}: T={ep['T']}, tau*={ep['tau_star']} "
              f"({ep['tau_star']/ep['T']*100:.1f}%)")
    print(f"  ...")

    # Compute and write temporal proportions
    proportions = compute_temporal_proportions(episodes)
    write_temporal_proportions(proportions)

    # Update episodes parquet
    update_episodes_parquet(episodes)

    # Verify
    print("\nVerifying annotations...")
    verify(episodes)

    print("\nDone! SARM annotations ready for training.")


if __name__ == "__main__":
    main()
