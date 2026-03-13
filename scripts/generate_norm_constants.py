"""
Generate Normalization Constants
从 meta/stats.json 生成归一化常量 Python 文件

用法: python3 generate_norm_constants.py
"""

import json
from pathlib import Path


DATA_DIR = Path("/workspace/Mult-skill ACT/data")
OUTPUT_DIR = Path("/workspace/Mult-skill ACT/scripts/normalization")

# 数据集列表: (名称, stats.json 路径)
DATASETS = [
    ("all_points_v30", DATA_DIR / "all_points_v30" / "meta" / "stats.json"),
]
# 添加 point1~point10
for i in range(1, 11):
    DATASETS.append((
        f"point{i}_v30",
        DATA_DIR / "each-point-V30" / f"point{i}_v30" / "meta" / "stats.json",
    ))


def format_array(values, indent=4):
    """将数组格式化为多行 numpy 字符串"""
    lines = []
    for i, v in enumerate(values):
        comma = "," if i < len(values) - 1 else ""
        lines.append(f"{' ' * indent}{v}{comma}")
    return "\n".join(lines)


def generate_norm_file(name, stats_path, output_path):
    """读取 stats.json 并生成归一化常量 .py 文件"""
    with open(stats_path) as f:
        stats = json.load(f)

    state_mean = stats["observation.state"]["mean"]
    state_std = stats["observation.state"]["std"]
    action_mean = stats["action"]["mean"]
    action_std = stats["action"]["std"]

    content = f'''"""
归一化常量 - {name}
从 {stats_path.relative_to(DATA_DIR.parent)} 自动生成
"""
import numpy as np

# ====== 归一化统计量（从 {name} 数据集提取）======

# State 归一化参数 (observation.state) - {len(state_mean)}D
STATE_MEAN = np.array([
{format_array(state_mean)}
], dtype=np.float32)

STATE_STD = np.array([
{format_array(state_std)}
], dtype=np.float32)

# Action 反归一化参数 - {len(action_mean)}D (left_arm 7D + right_arm 7D)
ACTION_MEAN = np.array([
{format_array(action_mean)}
], dtype=np.float32)

ACTION_STD = np.array([
{format_array(action_std)}
], dtype=np.float32)

# Image 归一化参数（ImageNet 标准）
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
'''
    output_path.write_text(content)
    print(f"  Generated: {output_path.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Generating {len(DATASETS)} normalization files...\n")

    for name, stats_path in DATASETS:
        if not stats_path.exists():
            print(f"  SKIP: {stats_path} not found")
            continue
        output_file = OUTPUT_DIR / f"{name}_norm.py"
        generate_norm_file(name, stats_path, output_file)

    print(f"\nDone! {len(DATASETS)} files generated in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
