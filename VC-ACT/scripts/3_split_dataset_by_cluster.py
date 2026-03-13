"""
VC-ACT 阶段 3: 按聚类结果拆分 bag 文件并转换为 LeRobot 数据集

读取聚类标签，将 bag 文件按簇分组:
  1. 为每个簇创建子目录，软链接对应的 bag 文件
  2. 对每个簇子目录调用 bag2lerobot_v30.py 转换为 LeRobot V3.0 数据集
  3. 自动生成训练 yaml 配置

输入:
  - results/cluster_labels_k10.npy 或 cluster_labels_auto.npy
  - results/episode_meta.json
  - bag_dir/ (原始 bag 文件目录)

输出:
  - data/vc_k10/bags/cluster0/ ~ cluster9/     (bag 文件软链接)
  - data/vc_k10/cluster0_v30/ ~ cluster9_v30/  (LeRobot V3.0 数据集)
  - config/exp_k10/cluster0.yaml ~ cluster9.yaml

用法:
  # 实验二 (固定 k=10)
  /venv/mult-act/bin/python3 3_split_dataset_by_cluster.py --mode k10

  # 实验三 (自动 k)
  /venv/mult-act/bin/python3 3_split_dataset_by_cluster.py --mode auto
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MAIN_ROOT = PROJECT_ROOT.parent
BAG2LEROBOT_SCRIPT = MAIN_ROOT / "scripts" / "bag2lerobot_v30.py"
PYTHON = "/venv/mult-act/bin/python3"

WANDB_KEY = "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"


def generate_yaml(cluster_id: int, exp_name: str, dataset_id: str,
                  dataset_root: str, output_root: str, gpu_id: int = 0) -> str:
    """Generate training yaml config matching exp2_point1.yaml format."""
    run_name = f"{exp_name}_cluster{cluster_id}"
    return f"""# VC-ACT {exp_name}: Cluster {cluster_id}
# /venv/mult-act/bin/python3 /workspace/Mult-skill\\ ACT/train/run_train.py --config <this_file>

runtime:
  command: "/venv/mult-act/bin/lerobot-train"
  check_command: true
  cwd: "."
  dry_run: false
  env:
    CUDA_VISIBLE_DEVICES: "{gpu_id}"
    WANDB_API_KEY: "{WANDB_KEY}"

variables:
  dataset_id: "{dataset_id}"
  dataset_root: "{dataset_root}"
  output_root: "{output_root}"
  run_name: "{run_name}"

train_args:
  dataset:
    repo_id: "{{dataset_id}}"
    root: "{{dataset_root}}/{{dataset_id}}"
    video_backend: "torchcodec"

  policy:
    type: "act"
    device: "cuda"
    repo_id: "local/{{run_name}}"
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

  output_dir: "{{output_root}}/{{run_name}}"

flags: []
raw_args: []
"""


def main(args):
    vc_root = Path(args.vc_root)
    results_dir = vc_root / "results"
    bag_dir = Path(args.bag_dir)

    # Load labels and metadata
    if args.mode == "k10":
        labels = np.load(results_dir / "cluster_labels_k10.npy")
        exp_name = "vc_k10"
    else:
        labels = np.load(results_dir / "cluster_labels_auto.npy")
        exp_name = "vc_auto"

    with open(results_dir / "episode_meta.json") as f:
        meta = json.load(f)

    k = int(labels.max()) + 1
    print(f"Mode: {args.mode}, k={k}")
    print(f"Label distribution: {np.bincount(labels).tolist()}")

    # Paths
    data_output_dir = Path(args.data_output_dir) / exp_name  # e.g. data/vc_k10/
    bags_split_dir = vc_root / "data" / exp_name / "bags"
    dataset_dir = data_output_dir  # LeRobot datasets go here
    config_dir = vc_root / "config" / f"exp_{args.mode}"
    models_dir = vc_root / "models" / f"exp_{args.mode}"

    for d in [bags_split_dir, config_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # === Step 1: Split bags into cluster subdirectories ===
    print("\n=== Step 1: Splitting bags by cluster ===")
    cluster_bag_mapping = {}

    for cluster_id in range(k):
        cluster_bag_dir = bags_split_dir / f"cluster{cluster_id}"
        if cluster_bag_dir.exists():
            shutil.rmtree(cluster_bag_dir)
        cluster_bag_dir.mkdir(parents=True)

        bag_indices = np.where(labels == cluster_id)[0]
        bag_names = [meta[i]["bag_name"] for i in bag_indices]
        cluster_bag_mapping[f"cluster{cluster_id}"] = bag_names

        for bag_name in bag_names:
            src = bag_dir / bag_name
            dst = cluster_bag_dir / bag_name
            if src.exists():
                os.symlink(src.resolve(), dst)
            else:
                print(f"  WARNING: {src} not found!")

        print(f"  cluster {cluster_id}: {len(bag_names)} bags -> {cluster_bag_dir}")

    # Save mapping
    mapping_path = results_dir / f"cluster_bag_mapping_{args.mode}.json"
    with open(mapping_path, "w") as f:
        json.dump(cluster_bag_mapping, f, indent=2)
    print(f"Mapping saved: {mapping_path}")

    # === Step 2: Convert each cluster's bags to LeRobot V3.0 ===
    print("\n=== Step 2: Converting bags to LeRobot V3.0 ===")
    for cluster_id in range(k):
        cluster_bag_dir = bags_split_dir / f"cluster{cluster_id}"
        dataset_id = f"cluster{cluster_id}_v30"
        output_path = dataset_dir / dataset_id

        n_bags = len(list(cluster_bag_dir.glob("*.bag")))
        if n_bags == 0:
            print(f"  cluster {cluster_id}: no bags, skipping")
            continue

        # Remove existing dataset if any
        if output_path.exists():
            shutil.rmtree(output_path)

        print(f"\n  cluster {cluster_id}: converting {n_bags} bags -> {output_path}")
        cmd = [
            PYTHON, str(BAG2LEROBOT_SCRIPT),
            "--data-dir", str(cluster_bag_dir),
            "--output-dir", str(dataset_dir),
            "--repo-name", dataset_id,
            "--task", f"cluster {cluster_id} manipulation",
            "--img-size", "256",
        ]
        print(f"  cmd: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR converting cluster {cluster_id}:")
            print(result.stderr[-500:] if result.stderr else "no stderr")
        else:
            # Print last few lines of stdout
            lines = result.stdout.strip().split("\n")
            for line in lines[-5:]:
                print(f"    {line}")

    # === Step 3: Generate training configs ===
    print("\n=== Step 3: Generating training configs ===")
    for cluster_id in range(k):
        dataset_id = f"cluster{cluster_id}_v30"
        dataset_path = dataset_dir / dataset_id
        if not dataset_path.exists():
            print(f"  cluster {cluster_id}: dataset not found, skipping config")
            continue

        yaml_content = generate_yaml(
            cluster_id=cluster_id,
            exp_name=exp_name,
            dataset_id=dataset_id,
            dataset_root=str(dataset_dir),
            output_root=str(models_dir),
            gpu_id=cluster_id % 4,
        )
        yaml_path = config_dir / f"cluster{cluster_id}.yaml"
        yaml_path.write_text(yaml_content)
        print(f"  {yaml_path}")

    print("\nDone!")
    print(f"\nTo train all clusters:")
    print(f"  for i in $(seq 0 {k-1}); do")
    print(f"    /venv/mult-act/bin/python3 /workspace/Mult-skill\\ ACT/train/run_train.py \\")
    print(f"      --config {config_dir}/cluster$i.yaml")
    print(f"  done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VC-ACT: Split bags by cluster and convert to LeRobot")
    parser.add_argument("--mode", type=str, choices=["k10", "auto"], required=True,
                        help="k10 = fixed 10 clusters, auto = optimal k from silhouette")
    parser.add_argument("--vc_root", type=str,
                        default="/workspace/Mult-skill ACT/VC-ACT")
    parser.add_argument("--data_output_dir", type=str,
                        default="/workspace/Mult-skill ACT/data",
                        help="Where to store converted LeRobot datasets")
    parser.add_argument("--bag_dir", type=str,
                        default="/workspace/Mult-skill ACT/data/26-03-04-pick_hanger")
    args = parser.parse_args()
    main(args)
