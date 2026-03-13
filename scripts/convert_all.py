"""
One-click conversion: 26-03-01-pick bags -> LeRobot V3.0 datasets
=================================================================
Generates two types of datasets:

Type A: 10 independent per-point datasets (for Experiment 2)
    data/each-point-V30/point1_v30/  ... data/each-point-V30/point10_v30/

Type B: 1 merged dataset with all 150 episodes (for Experiment 1 & 3)
    data/all_points_v30/
    Each episode labeled with task="pick towel from point N"

Usage:
    conda activate mult-act
    cd /workspace/Mult-skill\ ACT
    python scripts/convert_all.py
"""

import sys
import time
import re
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).parent))
from bag2lerobot_v30 import (
    process_single_bag, IMG_SIZE, ARM_DOF, FPS,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data" / "26-03-01-pick"
OUTPUT_DIR = PROJECT_ROOT / "data"
TYPE_A_DIR = OUTPUT_DIR / "each-point-V30"

NUM_WORKERS = min(32, cpu_count())


def get_features():
    """Return the standard feature dict for our dual-arm 4-cam setup."""
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (ARM_DOF * 2,),
            "names": [
                "left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3",
                "left_joint_4", "left_joint_5", "left_joint_6",
                "right_joint_0", "right_joint_1", "right_joint_2", "right_joint_3",
                "right_joint_4", "right_joint_5", "right_joint_6",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (ARM_DOF * 2,),
            "names": [
                "left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3",
                "left_joint_4", "left_joint_5", "left_joint_6",
                "right_joint_0", "right_joint_1", "right_joint_2", "right_joint_3",
                "right_joint_4", "right_joint_5", "right_joint_6",
            ],
        },
        "observation.images.fisheye_left": {
            "dtype": "video",
            "shape": (IMG_SIZE[1], IMG_SIZE[0], 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.fisheye_right": {
            "dtype": "video",
            "shape": (IMG_SIZE[1], IMG_SIZE[0], 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wide_mid": {
            "dtype": "video",
            "shape": (IMG_SIZE[1], IMG_SIZE[0], 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wide_top": {
            "dtype": "video",
            "shape": (IMG_SIZE[1], IMG_SIZE[0], 3),
            "names": ["height", "width", "channels"],
        },
    }


def extract_point_number(folder_name):
    """Extract point number from folder name like 'point1', 'point10'."""
    match = re.search(r'point(\d+)', folder_name)
    return int(match.group(1)) if match else None


def create_dataset(repo_name, output_dir):
    """Create a fresh LeRobotDataset."""
    output_path = output_dir / repo_name
    return LeRobotDataset.create(
        repo_id=repo_name,
        root=output_path,
        robot_type="dual_arm",
        fps=FPS,
        features=get_features(),
        use_videos=True,
        image_writer_threads=32,
        image_writer_processes=16,
        vcodec="h264_nvenc",
    )


def _process_bag_worker(args):
    """Worker function for parallel bag processing."""
    bag_path, bag_idx, total, task_label = args
    return process_single_bag(bag_path, bag_idx, total, task_label=task_label)


def convert_point_bags(point_folder, point_num, dataset, task_label):
    """Convert all bags in a point folder into the given dataset (parallel read, serial write)."""
    bag_files = sorted(point_folder.glob("*.bag"))
    total = len(bag_files)

    # Parallel: read and decode all bags at once
    worker_args = [
        (bag_path, idx, total, task_label)
        for idx, bag_path in enumerate(bag_files, 1)
    ]

    print(f"    [Point {point_num}] Reading {total} bags with {NUM_WORKERS} workers...")
    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(_process_bag_worker, worker_args)

    # Serial: write frames into dataset (not thread-safe)
    success = 0
    for idx, result in enumerate(results, 1):
        if result is not None:
            for frame in result:
                dataset.add_frame(frame)
            dataset.save_episode()
            success += 1
            print(f"    [Point {point_num}] Episode {idx}/{total} saved.")
        del result

    return success


def convert_type_a():
    """Type A: 10 independent per-point datasets."""
    print(f"\n{'#' * 60}")
    print(f"# TYPE A: Converting each point independently")
    print(f"{'#' * 60}")

    point_folders = sorted(
        [f for f in DATA_DIR.iterdir() if f.is_dir()],
        key=lambda x: extract_point_number(x.name) or 0
    )

    for folder in point_folders:
        point_num = extract_point_number(folder.name)
        if point_num is None:
            continue

        repo_name = f"point{point_num}_v30"
        task_label = f"pick towel from point {point_num}"
        print(f"\n--- Point {point_num}: {repo_name} ---")

        dataset = create_dataset(repo_name, TYPE_A_DIR)
        success = convert_point_bags(folder, point_num, dataset, task_label)
        dataset.finalize()

        print(f"  Point {point_num} done: {success} episodes saved to {TYPE_A_DIR / repo_name}")


def convert_type_b():
    """Type B: 1 merged dataset with all 150 episodes."""
    print(f"\n{'#' * 60}")
    print(f"# TYPE B: Converting all points into one merged dataset")
    print(f"{'#' * 60}")

    repo_name = "all_points_v30"
    dataset = create_dataset(repo_name, OUTPUT_DIR)

    point_folders = sorted(
        [f for f in DATA_DIR.iterdir() if f.is_dir()],
        key=lambda x: extract_point_number(x.name) or 0
    )

    total_success = 0
    for folder in point_folders:
        point_num = extract_point_number(folder.name)
        if point_num is None:
            continue

        task_label = f"pick towel from point {point_num}"
        print(f"\n--- Merging Point {point_num} ---")
        success = convert_point_bags(folder, point_num, dataset, task_label)
        total_success += success

    dataset.finalize()
    print(f"\n  Merged dataset done: {total_success} episodes saved to {OUTPUT_DIR / repo_name}")


def main():
    total_start = time.time()

    print(f"Data source: {DATA_DIR}")
    print(f"Output dir:  {OUTPUT_DIR}")
    print(f"FPS: {FPS}, Image: {IMG_SIZE}, State/Action: {ARM_DOF * 2}D")

    convert_type_a()
    convert_type_b()

    elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"All conversions complete! Total time: {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
