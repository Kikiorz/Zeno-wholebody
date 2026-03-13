"""
Upload Models to Hugging Face
上传所有模型（仅最新 checkpoint）到 HuggingFace

用法:
  python3 upload_models.py --repo_id QRP123/mult-skill-act-models --token hf_xxx
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo


MODELS_DIR = Path("/workspace/Mult-skill ACT/models")
NORM_DIR = Path("/workspace/Mult-skill ACT/scripts/normalization")


def upload_models(repo_id, token):
    api = HfApi(token=token)

    # 创建 repo（如果不存在）
    print(f"Creating/checking repo: {repo_id}")
    create_repo(repo_id, repo_type="model", exist_ok=True, token=token)

    # ========== 1. 上传 point_classifier ==========
    pc_dir = MODELS_DIR / "point_classifier"
    if pc_dir.exists():
        print(f"\n[1/4] Uploading point_classifier...")
        api.upload_folder(
            folder_path=str(pc_dir),
            path_in_repo="point_classifier",
            repo_id=repo_id,
            repo_type="model",
        )
        print("  Done: point_classifier/")

    # ========== 2. 上传 ACT 模型 (仅 last/pretrained_model) ==========
    act_models = sorted([
        d.name for d in MODELS_DIR.iterdir()
        if d.is_dir() and d.name != "point_classifier"
    ])

    print(f"\n[2/4] Uploading {len(act_models)} ACT models (last checkpoint only)...")
    for i, model_name in enumerate(act_models, 1):
        # 优先 last，否则取最大编号的 checkpoint
        last_dir = MODELS_DIR / model_name / "checkpoints" / "last" / "pretrained_model"
        if not last_dir.exists():
            # fallback: 找最大编号的 checkpoint
            ckpt_dir = MODELS_DIR / model_name / "checkpoints"
            if not ckpt_dir.exists():
                print(f"  SKIP: {model_name} (no checkpoints)")
                continue
            numbered = sorted([
                d for d in ckpt_dir.iterdir()
                if d.is_dir() and d.name.isdigit()
            ], key=lambda x: int(x.name))
            if numbered:
                last_dir = numbered[-1] / "pretrained_model"
            else:
                print(f"  SKIP: {model_name} (no valid checkpoint)")
                continue

        if not last_dir.exists():
            print(f"  SKIP: {model_name} (pretrained_model not found)")
            continue

        repo_path = f"{model_name}/pretrained_model"
        print(f"  [{i}/{len(act_models)}] {model_name} -> {repo_path}")
        api.upload_folder(
            folder_path=str(last_dir),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model",
        )

    # ========== 3. 上传归一化文件 ==========
    if NORM_DIR.exists() and any(NORM_DIR.glob("*.py")):
        print(f"\n[3/4] Uploading normalization files...")
        api.upload_folder(
            folder_path=str(NORM_DIR),
            path_in_repo="normalization",
            repo_id=repo_id,
            repo_type="model",
        )
        print("  Done: normalization/")
    else:
        print(f"\n[3/4] SKIP: No normalization files found at {NORM_DIR}")
        print("  Run generate_norm_constants.py first")

    # ========== 4. 创建 README ==========
    print(f"\n[4/4] Creating README...")
    readme = """# Mult-skill ACT Models

Multi-skill Action Chunking Transformer models for Piper dual-arm robot.

## Models

| Model | Description |
|-------|-------------|
| `point_classifier/` | ResNet-18 point classifier (10 classes: point1-point10) |
| `exp1_all_points/` | ACT policy trained on all 150 episodes (10 points combined) |
| `exp2_point1/` ~ `exp2_point10/` | ACT policies trained per-point (15 episodes each) |
| `exp3_act_x/` | Experimental ACT policy |

## Normalization

`normalization/` contains pre-computed normalization constants for each dataset:
- `all_points_v30_norm.py` - Stats from combined dataset
- `point1_v30_norm.py` ~ `point10_v30_norm.py` - Stats from per-point datasets

## Usage

### Download all models
```bash
pip install huggingface_hub
huggingface-cli download {repo_id} --repo-type model --local-dir ./models
```

### Point Classifier Inference
```python
from point_classifier_model import PointClassifier
import torch

checkpoint = torch.load("point_classifier/point_classifier_best.pth")
model = PointClassifier(num_classes=10, pretrained=False)
model.load_state_dict(checkpoint["model_state_dict"])
```

## Robot Setup
- Robot: Piper dual-arm (7D + 7D = 14D state/action)
- Cameras: fisheye_left, fisheye_right, wide_mid, wide_top (256x256)
- FPS: 30
"""
    readme = readme.replace("{repo_id}", repo_id)
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print("  Done: README.md")

    print(f"\nAll uploads complete!")
    print(f"View at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload models to Hugging Face")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="HuggingFace repo ID (e.g. QRP123/mult-skill-act-models)")
    parser.add_argument("--token", type=str, required=True,
                        help="HuggingFace write token")
    args = parser.parse_args()
    upload_models(args.repo_id, args.token)


if __name__ == "__main__":
    main()
