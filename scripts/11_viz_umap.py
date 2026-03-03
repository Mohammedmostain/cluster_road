import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

ROOT = Path(__file__).resolve().parent.parent

# =========================
# CONFIG
# =========================
WORK_DIR = ROOT / "work"

LABELS_CSV = WORK_DIR / "multimodal_hdbscan" / "labels.csv"     # filename,cluster,probability (or without prob)
EMB_PATH   = WORK_DIR / "embeddings.npy"
FN_PATH    = WORK_DIR / "filenames.txt"

OUT_PNG = WORK_DIR / "viz_hdbscan_umap2d.png"

# Preprocess for visualization
DO_PCA = True
PCA_N = 50

# UMAP settings (2D)
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.05
UMAP_METRIC = "euclidean"

# =========================
# MAIN
# =========================
def main():
    labels_df = pd.read_csv(LABELS_CSV)
    emb = np.load(EMB_PATH)
    fns = [x.strip() for x in open(FN_PATH, "r", encoding="utf-8") if x.strip()]
    assert emb.shape[0] == len(fns), "embeddings.npy rows must match filenames.txt"

    # Build filename -> embedding row index
    fn_to_idx = {fn: i for i, fn in enumerate(fns)}

    # Keep only labels that exist in embeddings
    keep = labels_df["filename"].isin(fn_to_idx.keys())
    labels_df = labels_df[keep].copy()

    idxs = [fn_to_idx[fn] for fn in labels_df["filename"].tolist()]
    X = emb[idxs, :]

    # Standardize
    X = StandardScaler().fit_transform(X)

    # PCA
    if DO_PCA:
        X = PCA(n_components=PCA_N, random_state=42).fit_transform(X)

    # UMAP -> 2D
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=42,
    )
    X2 = reducer.fit_transform(X)

    labels = labels_df["cluster"].astype(int).to_numpy()

    # Optional probability column
    probs = None
    if "probability" in labels_df.columns:
        probs = pd.to_numeric(labels_df["probability"], errors="coerce").to_numpy()

    # Plot
    plt.figure(figsize=(10, 7))

    # Noise first (gray)
    noise_mask = labels == -1
    if noise_mask.any():
        plt.scatter(X2[noise_mask, 0], X2[noise_mask, 1], s=12, alpha=0.35, label="Noise (-1)")

    # Clusters
    for c in sorted(set(labels)):
        if c == -1:
            continue
        m = labels == c

        # If probabilities exist, scale point sizes slightly by confidence
        if probs is not None:
            p = probs[m]
            sizes = 12 + 70 * np.nan_to_num(p, nan=0.0)
            plt.scatter(X2[m, 0], X2[m, 1], s=sizes, alpha=0.75, label=f"Cluster {c}")
        else:
            plt.scatter(X2[m, 0], X2[m, 1], s=18, alpha=0.75, label=f"Cluster {c}")

    plt.title("HDBSCAN Clusters (UMAP 2D projection)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(markerscale=1.2, fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    plt.close()

    print("Saved:", OUT_PNG)

if __name__ == "__main__":
    main()
