import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, f_oneway, kruskal
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# CONFIG
# =========================
WORK_DIR = Path("work")

# Pick ONE at a time (or run twice with different values)
METHOD = "hdbscan"   # "kmeans" or "hdbscan" or "dbscan"

CLUSTER_LABELS_PATH = {
    "kmeans":  WORK_DIR / "kmeans_labels.csv",
    "hdbscan": WORK_DIR / "hdbscan_labels.csv",
    "dbscan":  WORK_DIR / "dbscan_labels.csv",   # if you have it
}[METHOD]

TRAFFIC_CSV_PATH = Path("traffic_data.csv")  # <-- change if needed

# Remove noise cluster -1 (common in DBSCAN/HDBSCAN)
REMOVE_NOISE = METHOD in {"hdbscan", "dbscan"}

# =========================
# OUTPUTS (organized)
# =========================
OUT_DIR = WORK_DIR / f"analysis_{METHOD}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MERGED_CSV   = OUT_DIR / f"merged_clusters_traffic_{METHOD}.csv"
OUT_STATS_CSV    = OUT_DIR / f"cluster_stats_{METHOD}.csv"
OUT_RANKMAP_CSV  = OUT_DIR / f"cluster_rank_map_{METHOD}.csv"
OUT_REPORT_TXT   = OUT_DIR / f"cluster_aawdt_report_{METHOD}.txt"
OUT_BOXPLOT      = OUT_DIR / f"aawdt_by_cluster_{METHOD}.png"
OUT_SCATTER      = OUT_DIR / f"aawdt_vs_cluster_rank_{METHOD}.png"


def main():
    # --- Load files ---
    clusters = pd.read_csv(CLUSTER_LABELS_PATH)
    traffic = pd.read_csv(TRAFFIC_CSV_PATH)

    # Validate expected columns early
    required_cluster_cols = {"filename", "cluster"}
    missing = required_cluster_cols - set(clusters.columns)
    if missing:
        raise ValueError(f"Cluster labels file is missing columns: {missing}. "
                         f"Expected at least {required_cluster_cols}")

    if "Estimation_point" not in traffic.columns:
        raise ValueError("Traffic CSV must contain an 'Estimation_point' column.")
    if "AAWDT" not in traffic.columns:
        raise ValueError("Traffic CSV must contain an 'AAWDT' column.")

    # --- Normalize keys ---
    clusters["Estimation_point"] = (
        clusters["filename"]
        .astype(str)
        .str.replace(".png", "", regex=False)
        .str.replace(".jpg", "", regex=False)
        .str.replace(".jpeg", "", regex=False)
    )

    clusters["Estimation_point"] = pd.to_numeric(clusters["Estimation_point"], errors="coerce")
    traffic["Estimation_point"] = pd.to_numeric(traffic["Estimation_point"], errors="coerce")

    clusters = clusters.dropna(subset=["Estimation_point"])
    traffic = traffic.dropna(subset=["Estimation_point"])

    clusters["Estimation_point"] = clusters["Estimation_point"].astype(int)
    traffic["Estimation_point"] = traffic["Estimation_point"].astype(int)

    # --- Merge ---
    merged = pd.merge(
        clusters[["Estimation_point", "filename", "cluster"]],
        traffic,
        on="Estimation_point",
        how="inner"
    )

    if REMOVE_NOISE and "cluster" in merged.columns:
        merged = merged[merged["cluster"] != -1]

    if merged.empty:
        raise RuntimeError(
            "Merged dataframe is empty. Check that filename IDs match Estimation_point values."
        )

    merged.to_csv(OUT_MERGED_CSV, index=False)

    # --- Per-cluster stats ---
    stats_tbl = (
        merged.groupby("cluster")["AAWDT"]
        .agg(count="count", mean="mean", median="median", std="std", min="min", max="max")
        .sort_index()
    )
    stats_tbl.to_csv(OUT_STATS_CSV)

    # --- ANOVA + Kruskal (only if 2+ clusters with data) ---
    groups = [g["AAWDT"].values for _, g in merged.groupby("cluster")]
    do_tests = len(groups) >= 2 and all(len(arr) > 0 for arr in groups)

    if do_tests:
        anova_res = f_oneway(*groups)
        kruskal_res = kruskal(*groups)
    else:
        anova_res = None
        kruskal_res = None

    # --- Eta-squared ---
    grand_mean = merged["AAWDT"].mean()
    ss_between = sum(
        len(g) * (g["AAWDT"].mean() - grand_mean) ** 2
        for _, g in merged.groupby("cluster")
    )
    ss_total = ((merged["AAWDT"] - grand_mean) ** 2).sum()
    eta_sq = float(ss_between / ss_total) if ss_total != 0 else float("nan")

    # --- Spearman correlation with rank-by-median ---
    medians = stats_tbl["median"].sort_values()  # low -> high
    rank_map = {cluster_id: rank + 1 for rank, cluster_id in enumerate(medians.index.tolist())}
    merged["cluster_rank"] = merged["cluster"].map(rank_map)

    rankmap_df = pd.DataFrame(
        {"cluster": list(rank_map.keys()), "rank": list(rank_map.values())}
    ).sort_values("rank")
    rankmap_df.to_csv(OUT_RANKMAP_CSV, index=False)

    rho, rho_p = spearmanr(merged["cluster_rank"], merged["AAWDT"])

    # --- Report ---
    with open(OUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(f"AAWDT vs Image Clusters ({METHOD.upper()})\n")
        f.write("================================\n\n")

        f.write(f"Cluster labels file: {CLUSTER_LABELS_PATH}\n")
        f.write(f"Traffic CSV: {TRAFFIC_CSV_PATH}\n")
        f.write(f"Noise removed (-1): {REMOVE_NOISE}\n\n")

        f.write("Per-cluster AAWDT stats:\n")
        f.write(stats_tbl.to_string(float_format=lambda x: f"{x:,.3f}"))
        f.write("\n\n")

        f.write("Cluster rank mapping (by median AAWDT, low→high):\n")
        for _, row in rankmap_df.iterrows():
            f.write(f"  Cluster {int(row['cluster'])} -> Rank {int(row['rank'])}\n")
        f.write("\n")

        if do_tests:
            f.write(f"ANOVA p-value: {anova_res.pvalue:.6e}\n")
            f.write(f"Kruskal-Wallis p-value: {kruskal_res.pvalue:.6e}\n")
        else:
            f.write("ANOVA/Kruskal not run (need at least 2 clusters with data).\n")

        f.write(f"Effect size (eta squared): {eta_sq:.4f}\n\n")

        f.write("Spearman correlation (cluster_rank vs AAWDT):\n")
        f.write(f"  rho: {rho:.4f}\n")
        f.write(f"  p-value: {rho_p:.6e}\n")

    # --- Plots ---
    sns.set_context("talk")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=merged, x="cluster", y="AAWDT")
    plt.title(f"AAWDT Distribution by Image Cluster ({METHOD.upper()})")
    plt.xlabel("Cluster")
    plt.ylabel("AAWDT")
    plt.tight_layout()
    plt.savefig(OUT_BOXPLOT, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    x = merged["cluster_rank"].astype(float)
    jitter = np.random.uniform(-0.08, 0.08, size=len(x))
    plt.scatter(x + jitter, merged["AAWDT"], alpha=0.35)
    plt.title(f"AAWDT vs Cluster Rank ({METHOD.upper()}), Spearman rho={rho:.2f}")
    plt.xlabel("Cluster Rank (low → high by median AAWDT)")
    plt.ylabel("AAWDT")
    plt.xticks(sorted(merged["cluster_rank"].unique()))
    plt.tight_layout()
    plt.savefig(OUT_SCATTER, dpi=200)
    plt.close()

    # --- Console summary ---
    print("\n✅ Outputs saved to:", OUT_DIR.resolve())
    print("✅ Merged:", OUT_MERGED_CSV)
    print("✅ Stats:", OUT_STATS_CSV)
    print("✅ Rank map:", OUT_RANKMAP_CSV)
    print("✅ Report:", OUT_REPORT_TXT)
    print("✅ Plots:", OUT_BOXPLOT, "and", OUT_SCATTER)

    print("\n--- Quick Results ---")
    print(stats_tbl)
    print("\nRank map (low→high):", rank_map)
    print(f"\nSpearman rho={rho:.4f}, p={rho_p:.3e}")
    print(f"Eta^2={eta_sq:.4f}")
    if do_tests:
        print(f"ANOVA p={anova_res.pvalue:.3e}, Kruskal p={kruskal_res.pvalue:.3e}")


if __name__ == "__main__":
    main()
