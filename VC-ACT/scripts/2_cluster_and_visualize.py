"""
VC-ACT 阶段 2: 聚类与可视化

读取阶段 1 输出的 (N, 2048) 特征，执行:
  1. K-Means 聚类 (固定 k=10 和 自动搜索最优 k)
  2. Silhouette score 选最优 k (k=2..max_k)
  3. t-SNE 可视化 (按聚类标签着色 + 按 bag 编号着色)
  4. 每个簇包含哪些 bag 的详细分布

输出:
  - results/cluster_labels_k10.npy        (N,) int, k=10 聚类标签
  - results/cluster_labels_auto.npy       (N,) int, 最优 k 聚类标签
  - results/cluster_centers_k10.npy       (10, 2048) 簇中心
  - results/cluster_centers_auto.npy      (k*, 2048) 簇中心
  - results/optimal_k.json               {"best_k": k*, ...}
  - results/tsne_k10.png                 t-SNE (k=10 聚类着色)
  - results/tsne_auto.png                t-SNE (auto k 聚类着色)
  - results/silhouette_scores.png        Silhouette score 曲线

用法:
  /venv/mult-act/bin/python3 2_cluster_and_visualize.py
  /venv/mult-act/bin/python3 2_cluster_and_visualize.py --max_k 15
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_kmeans(features: np.ndarray, k: int, seed: int = 42) -> tuple:
    """Run K-Means and return (labels, centers, inertia)."""
    km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
    labels = km.fit_predict(features)
    return labels, km.cluster_centers_, km.inertia_


def find_optimal_k(features: np.ndarray, max_k: int, seed: int = 42) -> tuple:
    """Compute silhouette scores for k=2..max_k, return results dict and best k."""
    results = {}
    for k in range(2, max_k + 1):
        labels, _, inertia = run_kmeans(features, k, seed)
        score = silhouette_score(features, labels)
        results[k] = {"silhouette": score, "inertia": inertia}
        print(f"  k={k:2d}: silhouette={score:.4f}, inertia={inertia:.1f}")
    best_k = max(results, key=lambda k: results[k]["silhouette"])
    return results, best_k


def plot_silhouette(scores_dict: dict, best_k: int, save_path: str):
    """Plot silhouette scores vs k."""
    ks = sorted(scores_dict.keys())
    sils = [scores_dict[k]["silhouette"] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, sils, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.axvline(best_k, color="red", linestyle="--", alpha=0.7, label=f"best k={best_k}")
    ax.set_xlabel("Number of clusters (k)", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_title("Silhouette Score vs K", fontsize=14)
    ax.set_xticks(ks)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved silhouette plot to {save_path}")


def plot_tsne(features: np.ndarray, cluster_labels: np.ndarray, bag_names: list,
              k_value: int, save_path: str, seed: int = 42):
    """Plot t-SNE colored by cluster labels, annotated with bag names."""
    tsne = TSNE(n_components=2, random_state=seed, perplexity=min(30, len(features) - 1))
    coords = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(12, 9))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=cluster_labels,
                         cmap="tab10", s=60, alpha=0.8, edgecolors="k", linewidths=0.3)
    ax.set_title(f"t-SNE Visualization (k={k_value})", fontsize=14)
    ax.legend(*scatter.legend_elements(), title="Cluster", loc="best", fontsize=9)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.2)

    # Annotate a subset of points with bag names to avoid clutter
    # Only annotate every N-th point if too many
    step = max(1, len(bag_names) // 30)
    for i in range(0, len(bag_names), step):
        short_name = bag_names[i].replace("pick_hanger_", "").replace(".bag", "")
        ax.annotate(short_name, (coords[i, 0], coords[i, 1]),
                    fontsize=6, alpha=0.6, ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved t-SNE plot to {save_path}")


def main(args):
    results_dir = Path(args.results_dir)

    # Load features and metadata
    features = np.load(results_dir / "features_2048.npy")
    with open(results_dir / "episode_meta.json") as f:
        meta = json.load(f)
    bag_names = [m["bag_name"] for m in meta]
    print(f"Loaded features: {features.shape}, {len(bag_names)} bags")

    # Standardize features for better clustering
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # === 1. Fixed k=10 ===
    print("\n=== K-Means k=10 ===")
    labels_k10, centers_k10, _ = run_kmeans(features_scaled, k=10, seed=args.seed)
    np.save(results_dir / "cluster_labels_k10.npy", labels_k10)
    np.save(results_dir / "cluster_centers_k10.npy", centers_k10)
    sil_k10 = silhouette_score(features_scaled, labels_k10)
    print(f"k=10 silhouette score: {sil_k10:.4f}")

    # === 2. Auto k selection ===
    print(f"\n=== Silhouette search k=2..{args.max_k} ===")
    scores_dict, best_k = find_optimal_k(features_scaled, args.max_k, args.seed)
    print(f"\nOptimal k = {best_k} (silhouette={scores_dict[best_k]['silhouette']:.4f})")

    labels_auto, centers_auto, _ = run_kmeans(features_scaled, k=best_k, seed=args.seed)
    np.save(results_dir / "cluster_labels_auto.npy", labels_auto)
    np.save(results_dir / "cluster_centers_auto.npy", centers_auto)

    with open(results_dir / "optimal_k.json", "w") as f:
        json.dump({
            "best_k": best_k,
            "silhouette_score": scores_dict[best_k]["silhouette"],
            "all_scores": {str(k): v for k, v in scores_dict.items()},
        }, f, indent=2)

    # === 3. Visualizations ===
    print("\n=== Generating plots ===")

    # Silhouette scores
    plot_silhouette(scores_dict, best_k, str(results_dir / "silhouette_scores.png"))

    # t-SNE for k=10
    plot_tsne(features_scaled, labels_k10, bag_names, 10,
              str(results_dir / "tsne_k10.png"), seed=args.seed)

    # t-SNE for auto k
    if best_k != 10:
        plot_tsne(features_scaled, labels_auto, bag_names, best_k,
                  str(results_dir / "tsne_auto.png"), seed=args.seed)

    # === 4. Summary ===
    print("\n=== Cluster distribution (k=10) ===")
    for c in range(10):
        bag_indices = np.where(labels_k10 == c)[0]
        names = [bag_names[i] for i in bag_indices]
        print(f"  cluster {c}: {len(bag_indices)} bags")
        for n in names:
            print(f"    - {n}")

    print(f"\n=== Cluster distribution (k={best_k}, auto) ===")
    for c in range(best_k):
        bag_indices = np.where(labels_auto == c)[0]
        names = [bag_names[i] for i in bag_indices]
        print(f"  cluster {c}: {len(bag_indices)} bags")
        for n in names:
            print(f"    - {n}")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VC-ACT: Cluster and visualize")
    parser.add_argument("--results_dir", type=str,
                        default="/workspace/Mult-skill ACT/VC-ACT/results")
    parser.add_argument("--max_k", type=int, default=10,
                        help="Maximum k to search (silhouette search from 2 to max_k)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
