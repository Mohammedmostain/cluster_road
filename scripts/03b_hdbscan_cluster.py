import os
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import hdbscan

# Optional (often improves HDBSCAN results)
USE_UMAP = True
if USE_UMAP:
    import umap

ROOT = Path(__file__).resolve().parent.parent

# =========================
# CONFIG
# =========================
WORK_DIR = ROOT / "work"

EMB_PATH = WORK_DIR / "embeddings.npy"
FN_PATH = WORK_DIR / "filenames.txt"
MANIFEST_PATH = WORK_DIR / "pairs_manifest.csv"

OUT_BASE = WORK_DIR / "clusters_hdbscan"
LABELS_CSV = WORK_DIR / "hdbscan_labels.csv"
METRICS_JSON = WORK_DIR / "hdbscan_metrics.json"

# Preprocessing
DO_PCA = True
PCA_VARIANCE = 0.95

# Optional UMAP (recommended if embeddings are high-dim & messy)
DO_UMAP = True
UMAP_N_COMPONENTS = 30
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.05
UMAP_METRIC = "euclidean"

# HDBSCAN params (tune these)
MIN_CLUSTER_SIZE = 20
MIN_SAMPLES = None          # None -> defaults to min_cluster_size
METRIC = "euclidean"
CLUSTER_SELECTION_METHOD = "eom"  # "eom" or "leaf"

EXPORT_MODE = "copy"   # or "symlink"


# =========================
# UTILS
# =========================
def read_filenames(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_manifest(manifest_csv: Path):
    mapping = {}
    with open(manifest_csv, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        i_fn = header.index("filename")
        i_in = header.index("zoom_in_path")
        i_out = header.index("zoom_out_path")

        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            mapping[parts[i_fn]] = (parts[i_in], parts[i_out])

    return mapping


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def export_file(src: str, dst: Path):
    ensure_dir(dst.parent)

    if EXPORT_MODE == "symlink":
        if not dst.exists():
            os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


# =========================
# MAIN
# =========================
def main():
    print("Loading embeddings...")
    emb = np.load(EMB_PATH)
    fns = read_filenames(FN_PATH)
    assert emb.shape[0] == len(fns), "Embeddings rows must match filenames"

    print("Embeddings shape:", emb.shape)

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(emb)

    # PCA (optional)
    pca_info = {"pca_used": False}
    if DO_PCA:
        pca = PCA(n_components=PCA_VARIANCE, random_state=42)
        X = pca.fit_transform(X)
        pca_info = {
            "pca_used": True,
            "n_components": int(X.shape[1]),
            "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        }
        print("After PCA:", X.shape)

    # UMAP (optional, but can help HDBSCAN a lot)
    umap_info = {"umap_used": False}
    if DO_UMAP:
        reducer = umap.UMAP(
            n_components=UMAP_N_COMPONENTS,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            metric=UMAP_METRIC,
            random_state=42,
        )
        X = reducer.fit_transform(X)
        umap_info = {
            "umap_used": True,
            "n_components": int(UMAP_N_COMPONENTS),
            "n_neighbors": int(UMAP_N_NEIGHBORS),
            "min_dist": float(UMAP_MIN_DIST),
            "metric": UMAP_METRIC,
        }
        print("After UMAP:", X.shape)

    print("Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric=METRIC,
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
    )

    labels = clusterer.fit_predict(X)  # noise = -1
    probs = getattr(clusterer, "probabilities_", None)

    n_noise = int(np.sum(labels == -1))
    unique = sorted(set(int(x) for x in labels))
    n_clusters = len([u for u in unique if u != -1])

    print(f"Clusters found: {n_clusters} (noise points: {n_noise})")
    print("Label set:", unique)

    # Save labels
    with open(LABELS_CSV, "w", encoding="utf-8") as f:
        f.write("filename,cluster,probability\n")
        for i, (fn, lab) in enumerate(zip(fns, labels)):
            p = float(probs[i]) if probs is not None else ""
            f.write(f"{fn},{int(lab)},{p}\n")

    # Export images
    mapping = load_manifest(MANIFEST_PATH)
    ensure_dir(OUT_BASE)

    for fn, lab in zip(fns, labels):
        if fn not in mapping:
            continue
        p_in, p_out = mapping[fn]

        # Put noise in its own folder
        cluster_name = "noise" if int(lab) == -1 else f"cluster_{int(lab)}"
        cluster_dir = OUT_BASE / cluster_name

        export_file(p_in, cluster_dir / "zoom_in" / fn)
        export_file(p_out, cluster_dir / "zoom_out" / fn)

    # Metrics
    sizes = {}
    for lab in labels:
        sizes[int(lab)] = sizes.get(int(lab), 0) + 1

    metrics = {
        "method": "hdbscan",
        "min_cluster_size": int(MIN_CLUSTER_SIZE),
        "min_samples": MIN_SAMPLES,
        "metric": METRIC,
        "cluster_selection_method": CLUSTER_SELECTION_METHOD,
        "num_clusters_excluding_noise": int(n_clusters),
        "num_noise": int(n_noise),
        "cluster_sizes": sizes,
        "pca": pca_info,
        "umap": umap_info,
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved:")
    print(" -", LABELS_CSV)
    print(" -", METRICS_JSON)
    print(" - Clusters exported to:", OUT_BASE)


if __name__ == "__main__":
    main()
