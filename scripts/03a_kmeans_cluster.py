import os
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent.parent

# =========================
# CONFIG
# =========================
WORK_DIR = ROOT / "work"

EMB_PATH = WORK_DIR / "embeddings.npy"
FN_PATH = WORK_DIR / "filenames.txt"
MANIFEST_PATH = WORK_DIR / "pairs_manifest.csv"

OUT_BASE = WORK_DIR / "clusters_kmeans"
LABELS_CSV = WORK_DIR / "kmeans_labels.csv"
METRICS_JSON = WORK_DIR / "kmeans_metrics.json"
ELBOW_PLOT = WORK_DIR / "kmeans_elbow.png"

DO_PCA = True
PCA_VARIANCE = 0.95

K_MIN = 2
K_MAX = 15

RANDOM_STATE = 42
N_INIT = 20

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
# ELBOW DETECTION
# =========================
def detect_elbow(inertias):
    """
    Simple knee detection:
    finds point of maximum second derivative-like change
    """
    diffs = np.diff(inertias)
    diff2 = np.diff(diffs)

    elbow_index = np.argmax(np.abs(diff2)) + 2
    return elbow_index


# =========================
# MAIN
# =========================
def main():
    print("Loading embeddings...")

    emb = np.load(EMB_PATH)
    fns = read_filenames(FN_PATH)

    assert emb.shape[0] == len(fns)

    print("Embeddings shape:", emb.shape)

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(emb)

    # PCA (optional)
    pca_info = {}
    if DO_PCA:
        pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
        X = pca.fit_transform(X)

        pca_info = {
            "pca_used": True,
            "n_components": int(X.shape[1]),
            "explained_variance": float(np.sum(pca.explained_variance_ratio_))
        }

        print("After PCA:", X.shape)

    # ===== ELBOW METHOD =====
    inertias = []
    silhouettes = []

    print("Computing elbow curve...")

    for k in range(K_MIN, K_MAX + 1):
        kmeans = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=N_INIT
        )

        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)

        # silhouette only valid if >1 cluster and <N
        try:
            sil = silhouette_score(X, labels)
        except:
            sil = None

        silhouettes.append(sil)

        print(f"K={k:2d} | inertia={kmeans.inertia_:.2f} | silhouette={sil}")

    # Save elbow plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(K_MIN, K_MAX + 1), inertias, marker="o")
    plt.title("K-Means Elbow Curve")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.savefig(ELBOW_PLOT)
    plt.close()

    # Auto-detect best K
    best_k = detect_elbow(inertias) + (K_MIN - 1)

    print("\nAuto-detected best K (elbow):", best_k)

    # ===== FINAL CLUSTERING =====
    final_kmeans = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=N_INIT
    )

    labels = final_kmeans.fit_predict(X)

    # Save labels CSV
    with open(LABELS_CSV, "w", encoding="utf-8") as f:
        f.write("filename,cluster\n")
        for fn, lab in zip(fns, labels):
            f.write(f"{fn},{lab}\n")

    # Export images
    mapping = load_manifest(MANIFEST_PATH)
    ensure_dir(OUT_BASE)

    for fn, lab in zip(fns, labels):
        if fn not in mapping:
            continue

        p_in, p_out = mapping[fn]

        cluster_dir = OUT_BASE / f"cluster_{lab}"

        export_file(p_in, cluster_dir / "zoom_in" / fn)
        export_file(p_out, cluster_dir / "zoom_out" / fn)

    # Metrics
    final_sil = silhouette_score(X, labels)

    sizes = {}
    for lab in labels:
        sizes[int(lab)] = sizes.get(int(lab), 0) + 1

    metrics = {
        "method": "kmeans",
        "best_k": int(best_k),
        "inertias": inertias,
        "silhouettes": silhouettes,
        "final_silhouette": float(final_sil),
        "cluster_sizes": sizes,
        "pca": pca_info
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved:")
    print(" -", LABELS_CSV)
    print(" -", METRICS_JSON)
    print(" -", ELBOW_PLOT)
    print(" - Clusters exported to:", OUT_BASE)


if __name__ == "__main__":
    main()
