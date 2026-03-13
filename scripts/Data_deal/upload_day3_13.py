"""
Upload stage1_act + stage2_act Models to Hugging Face
仓库: QRP123/day3_13 (model repo)

用法:
  python3 upload_day3_13.py
"""

from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_ID = "QRP123/day3_13"
TOKEN = ""

MODELS_ROOT = Path("/workspace/Zeno-wholebody/model")

MODELS = [
    {
        "name": "stage1_act",
        "checkpoints": ["040000"],
    },
    {
        "name": "stage2_act",
        "checkpoints": ["040000"],
    },
]


def main():
    api = HfApi(token=TOKEN)

    print(f"Creating repo: {REPO_ID}")
    create_repo(REPO_ID, repo_type="model", exist_ok=True, token=TOKEN)

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
            print(f"  Done: {model_name}/{ckpt}")

    # README
    print(f"\nCreating README...")
    readme = "# day3_13 - Zeno Wholebody ACT Models\n\n"
    readme += "Stage1 (hanger) + Stage2 (move) ACT policies for Zeno wholebody robot.\n\n"
    readme += "## Models\n\n"
    readme += "| Model | Task | Checkpoint |\n"
    readme += "|-------|------|------------|\n"
    readme += "| stage1_act | hanger | 040000 |\n"
    readme += "| stage2_act | move   | 040000 |\n\n"
    readme += "## Input/Output\n\n"
    readme += "- `observation.state` 17D: [base_vx, base_vy, base_omega, left_0..6, right_0..6]\n"
    readme += "- `observation.images.realsense_top/left/right` 224x224\n"
    readme += "- `action` 17D: [base_vx, base_vy, base_omega, left_0..6, right_0..6]\n\n"
    readme += "## Download\n\n"
    readme += "```bash\n"
    readme += f"huggingface-cli download {REPO_ID} --repo-type model --local-dir ./day3_13\n"
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
