import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, f_oneway
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
WORK_DIR = Path("work")

EMB_PATH = WORK_DIR / "embeddings.npy"
FN_PATH = WORK_DIR / "filenames.txt"
TRAFFIC_CSV = Path("traffic_data.csv")   # <-- change if needed

K_VALUES = range(3, 13)   # K = 4 to 12

DO_PCA = True
PCA_VARIANCE = 0.95


RANDOM_STATE = 42

OUT_SUMMARY = WORK_DIR / "k_sweep_summary.csv"
OUT_PLOT = WORK_DIR / "k_sweep_relationship.png"


# =========================
# Helper Functions
# =========================
def load_filenames(path):
    with open(path, "r") as f:
        return [line.strip() for line in f]


def eta_squared(df):
    grand_mean = df["AAWDT"].mean()
    ss_between = sum(
        len(g) * (g["AAWDT"].mean() - grand_mean) ** 2
        for _, g in df.groupby("cluster")
    )
    ss_total = ((df["AAWDT"] - grand_mean) ** 2).sum()
    return ss_between / ss_total


# =========================
# MAIN
# =========================
def main():

    print("Loading embeddings...")
    X = np.load(EMB_PATH)
    filenames = load_filenames(FN_PATH)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if DO_PCA:
        pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
        X = pca.fit_transform(X)
        print("PCA components:", X.shape[1])

    traffic = pd.read_csv(TRAFFIC_CSV)
    traffic["Estimation_point"] = pd.to_numeric(
        traffic["Estimation_point"], errors="coerce"
    ).astype("Int64")

    summary_rows = []

    for K in K_VALUES:
        print(f"\nRunning KMeans for K={K}")

        kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=20)
        labels = kmeans.fit_predict(X)

        cluster_df = pd.DataFrame({
            "filename": filenames,
            "cluster": labels
        })

        cluster_df["Estimation_point"] = (
            cluster_df["filename"]
            .str.replace(".png", "", regex=False)
            .astype("Int64")
        )

        merged = pd.merge(
            cluster_df,
            traffic[["Estimation_point", "AAWDT"]],
            on="Estimation_point",
            how="inner"
        ).dropna()

        if merged.empty:
            print("No merged rows. Skipping.")
            continue

        # ===== Rank clusters by median AAWDT =====
        stats_tbl = (
            merged.groupby("cluster")["AAWDT"]
            .agg(["median", "mean", "count"])
        )

        sorted_clusters = stats_tbl["median"].sort_values().index.tolist()

        rank_map = {
            cluster_id: rank + 1
            for rank, cluster_id in enumerate(sorted_clusters)
        }

        merged["cluster_rank"] = merged["cluster"].map(rank_map)

        # ===== Spearman =====
        rho, rho_p = spearmanr(
            merged["cluster_rank"],
            merged["AAWDT"]
        )

        # ===== ANOVA =====
        groups = [g["AAWDT"].values for _, g in merged.groupby("cluster")]
        anova_p = f_oneway(*groups).pvalue

        # ===== Effect size =====
        eta = eta_squared(merged)

        summary_rows.append({
            "K": K,
            "eta_squared": eta,
            "spearman_rho": rho,
            "spearman_p": rho_p,
            "anova_p": anova_p
        })

        print(f"K={K} | eta²={eta:.4f} | rho={rho:.4f}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, index=False)

    # ===== Plot relationship vs K =====
    plt.figure(figsize=(8, 5))
    plt.plot(summary["K"], summary["eta_squared"], marker="o", label="Eta-squared")
    plt.plot(summary["K"], summary["spearman_rho"], marker="o", label="Spearman rho")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Score")
    plt.title("Cluster–AAWDT Relationship vs K")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=200)
    plt.close()

    print("\nSaved summary:", OUT_SUMMARY)
    print("Saved plot:", OUT_PLOT)


if __name__ == "__main__":
    main()