import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import umap
import hdbscan

ROOT = Path(__file__).resolve().parent.parent

# =========================
# CONFIG
# =========================
WORK_DIR = ROOT / "work"

EMB_PATH = WORK_DIR / "embeddings.npy"
FN_PATH  = WORK_DIR / "filenames.txt"

META_NPY  = WORK_DIR / "metadata_features.npy"
MASK_NPY  = WORK_DIR / "metadata_match_mask.npy"
META_INFO = WORK_DIR / "metadata_info.json"

OUT_DIR = WORK_DIR / "multimodal_hdbscan"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_LABELS  = OUT_DIR / "labels.csv"
OUT_METRICS = OUT_DIR / "metrics.json"

# ---- Preprocess for density clustering ----
# For HDBSCAN, don't keep hundreds of PCA dims (like 285); use fixed low dim.
DO_PCA = True
PCA_N_COMPONENTS = 64

# ---- UMAP (recommended) ----
DO_UMAP = True
UMAP_N_COMPONENTS = 30
UMAP_N_NEIGHBORS  = 15
UMAP_MIN_DIST     = 0.05
UMAP_METRIC       = "euclidean"

# ---- HDBSCAN ----
MIN_CLUSTER_SIZE = 20
MIN_SAMPLES = None
HDBSCAN_METRIC = "euclidean"
CLUSTER_SELECTION_METHOD = "eom"

# ---- Modal weighting ----
# metadata is already standardized in your builder. Weight it so it doesn't dominate.
META_WEIGHT = 0.75


def load_lines(path: Path):
    return [x.strip() for x in open(path, "r", encoding="utf-8") if x.strip()]


def main():
    # Load image embeddings
    emb = np.load(EMB_PATH)
    fns = load_lines(FN_PATH)
    assert emb.shape[0] == len(fns), "embeddings.npy rows must match filenames.txt"

    # Load metadata + match mask
    X_meta = np.load(META_NPY).astype(np.float32)
    match_mask = np.load(MASK_NPY).astype(bool)

    assert X_meta.shape[0] == len(fns), "metadata_features.npy must align with filenames.txt"
    assert match_mask.shape[0] == len(fns), "metadata_match_mask.npy must align with filenames.txt"

    # Keep only matched rows (yours are all matched, but keep this for robustness)
    keep_idx = np.where(match_mask)[0]
    X_img = emb[keep_idx, :]
    X_meta = X_meta[keep_idx, :]
    kept_fns = [fns[i] for i in keep_idx]

    print("Rows kept:", len(kept_fns))

    # Scale image embeddings (important)
    X_img = StandardScaler().fit_transform(X_img).astype(np.float32)

    # Weight metadata
    X_meta = (X_meta * META_WEIGHT).astype(np.float32)

    # Concatenate
    X = np.concatenate([X_img, X_meta], axis=1)
    print("Combined feature shape:", X.shape)

    # PCA to fixed dim
    pca_info = {"used": False}
    if DO_PCA:
        pca = PCA(n_components=PCA_N_COMPONENTS, random_state=42)
        X = pca.fit_transform(X).astype(np.float32)
        pca_info = {"used": True, "n_components": int(PCA_N_COMPONENTS)}
        print("After PCA:", X.shape)

    # UMAP
    umap_info = {"used": False}
    if DO_UMAP:
        reducer = umap.UMAP(
            n_components=UMAP_N_COMPONENTS,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            metric=UMAP_METRIC,
            random_state=42,
        )
        X = reducer.fit_transform(X).astype(np.float32)
        umap_info = {
            "used": True,
            "n_components": int(UMAP_N_COMPONENTS),
            "n_neighbors": int(UMAP_N_NEIGHBORS),
            "min_dist": float(UMAP_MIN_DIST),
            "metric": UMAP_METRIC,
        }
        print("After UMAP:", X.shape)

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
    )

    labels = clusterer.fit_predict(X)
    probs = getattr(clusterer, "probabilities_", None)

    n_noise = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Save labels
    out = pd.DataFrame({
        "filename": kept_fns,
        "cluster": labels,
        "probability": probs if probs is not None else np.nan,
    })
    out.to_csv(OUT_LABELS, index=False)

    # Metrics
    sizes = {}
    for lab in labels:
        sizes[int(lab)] = sizes.get(int(lab), 0) + 1

    meta_cols = None
    if META_INFO.exists():
        meta_cols = json.loads(META_INFO.read_text(encoding="utf-8")).get("meta_cols")

    metrics = {
        "method": "hdbscan_multimodal",
        "rows": int(len(kept_fns)),
        "meta_cols": meta_cols,
        "meta_weight": float(META_WEIGHT),
        "pca": pca_info,
        "umap": umap_info,
        "hdbscan": {
            "min_cluster_size": int(MIN_CLUSTER_SIZE),
            "min_samples": MIN_SAMPLES,
            "metric": HDBSCAN_METRIC,
            "cluster_selection_method": CLUSTER_SELECTION_METHOD,
        },
        "num_clusters_excluding_noise": int(n_clusters),
        "num_noise": int(n_noise),
        "cluster_sizes": sizes,
    }

    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ Saved:", OUT_LABELS)
    print("✅ Saved:", OUT_METRICS)
    print(f"Clusters (excluding noise): {n_clusters}")
    print(f"Noise: {n_noise}/{len(labels)} ({100*n_noise/len(labels):.1f}%)")


if __name__ == "__main__":
    main()
