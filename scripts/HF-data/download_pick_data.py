#!/usr/bin/env python3
"""
从Hugging Face下载pick数据集的脚本
用法: python3 download_pick_data.py --repo_id username/pick-dataset
"""

import argparse
from huggingface_hub import snapshot_download

def download_pick_data(repo_id, local_dir="./pick_data"):
    """
    从Hugging Face下载pick数据集

    Args:
        repo_id: Hugging Face数据集ID (格式: username/dataset-name)
        local_dir: 本地保存目录
    """
    print(f"📦 开始下载数据集: {repo_id}")
    print(f"📂 保存到: {local_dir}")
    print("⏳ 下载中，请耐心等待...")

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir
        )
        print("\n✅ 下载完成！")
        print(f"📂 数据保存在: {local_dir}")
        print("\n📊 数据结构:")
        print("   point1/ - point10/")
        print("   每个文件夹包含约15个.bag文件")

    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n💡 建议:")
        print("1. 检查网络连接")
        print("2. 确认数据集ID是否正确")
        print("3. 如果是私有数据集，需要先登录: huggingface-cli login")
        print("4. 重新运行此命令会自动续传")
        raise

def main():
    parser = argparse.ArgumentParser(description="从Hugging Face下载pick数据集")
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face数据集ID (格式: username/dataset-name)"
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="./pick_data",
        help="本地保存目录（默认: ./pick_data）"
    )

    args = parser.parse_args()

    download_pick_data(
        repo_id=args.repo_id,
        local_dir=args.local_dir
    )

if __name__ == "__main__":
    main()
