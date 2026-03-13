"""
Upload day3_5_exp3 Model to Hugging Face
上传 moe_act_hanger_1300_V30 模型 + 归一化参数

用法:
  python3 upload_day3_5_exp3.py
"""

from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_ID = "QRP123/day3_5_exp3"
TOKEN = "YOUR_HF_TOKEN"

MODELS_ROOT = Path("/workspace/Mult-skill ACT/models")
NORM_DIR = Path("/workspace/Mult-skill ACT/scripts/normalization")

# 模型及其对应的归一化文件
MODELS = [
    {
        "name": "moe_act_hanger_1300_V30",
        "norm_file": "26-03-04-pick_hanger_V30_norm.py",
        "checkpoints": ["060000", "080000", "100010"],
    },
]


def main():
    api = HfApi(token=TOKEN)

    print(f"Creating repo: {REPO_ID}")
    create_repo(REPO_ID, repo_type="model", exist_ok=True, token=TOKEN)

    # 上传模型的 checkpoints
    for m_idx, model in enumerate(MODELS, 1):
        model_name = model["name"]
        print(f"\n[{m_idx}/{len(MODELS)}] Uploading model: {model_name}")

        for ckpt in model["checkpoints"]:
            ckpt_dir = MODELS_ROOT / model_name / "checkpoints" / ckpt / "pretrained_model"
            if not ckpt_dir.exists():
                print(f"  SKIP: {model_name}/{ckpt} not found")
                continue

            repo_path = f"{model_name}/checkpoints/{ckpt}/pretrained_model"
            print(f"  Uploading {ckpt} -> {repo_path}")
            api.upload_folder(
                folder_path=str(ckpt_dir),
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="model",
            )

    # 上传归一化文件
    print(f"\nUploading normalization files...")
    uploaded_norms = set()
    for model in MODELS:
        norm_name = model["norm_file"]
        if norm_name in uploaded_norms:
            continue
        norm_file = NORM_DIR / norm_name
        if norm_file.exists():
            api.upload_file(
                path_or_fileobj=str(norm_file),
                path_in_repo=f"normalization/{norm_name}",
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"  Uploaded {norm_name}")
            uploaded_norms.add(norm_name)
        else:
            print(f"  SKIP: {norm_name} not found")

    # README
    print(f"\nCreating README...")
    readme = "# day3_5_exp3 - MoE-ACT Hanger Model\n\n"
    readme += "## Model\n\n"
    readme += "| Model | Architecture | Dataset | Normalization |\n"
    readme += "|-------|-------------|---------|---------------|\n"
    readme += "| moe_act_hanger_1300_V30 | MoE-ACT | 26-03-04-pick_hanger_V30 | 26-03-04-pick_hanger_V30_norm.py |\n\n"
    readme += "## Checkpoints\n\n"
    readme += "Checkpoints at steps: 60000, 80000, 100010\n\n"
    readme += "## Usage\n\n"
    readme += "```bash\n"
    readme += f"huggingface-cli download {REPO_ID} --repo-type model --local-dir ./day3_5_exp3\n"
    readme += "```\n"

    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
    )
    print("  Done: README.md")

    print(f"\nUpload complete!")
    print(f"View at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
