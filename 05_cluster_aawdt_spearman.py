import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, f_oneway, kruskal
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# CONFIG (edit traffic csv path)
# =========================
WORK_DIR = Path("work")

CLUSTER_LABELS_PATH = WORK_DIR / "kmeans_labels.csv"   # filename,cluster
MANIFEST_PATH = WORK_DIR / "pairs_manifest.csv"        # optional, not required for this analysis

TRAFFIC_CSV_PATH = Path("traffic_data.csv")            # <-- CHANGE THIS to your csv filename

OUT_MERGED_CSV = WORK_DIR / "merged_clusters_traffic.csv"
OUT_REPORT_TXT = WORK_DIR / "cluster_aawdt_report.txt"
OUT_BOXPLOT = WORK_DIR / "aawdt_by_cluster.png"
OUT_SCATTER = WORK_DIR / "aawdt_vs_cluster_rank.png"


# =========================
# MAIN
# =========================
def main():
    # --- Load files ---
    clusters = pd.read_csv(CLUSTER_LABELS_PATH)
    traffic = pd.read_csv(TRAFFIC_CSV_PATH)

    # --- Normalize keys ---
    # cluster file has filename like "1182163.png" (or similar)
    # traffic file has Estimation_point as numeric id (as you said)
    clusters["Estimation_point"] = (
        clusters["filename"]
        .astype(str)
        .str.replace(".png", "", regex=False)
        .str.replace(".jpg", "", regex=False)
        .str.replace(".jpeg", "", regex=False)
    )

    # make numeric if possible
    clusters["Estimation_point"] = pd.to_numeric(clusters["Estimation_point"], errors="coerce")
    traffic["Estimation_point"] = pd.to_numeric(traffic["Estimation_point"], errors="coerce")

    # keep only rows with valid keys
    clusters = clusters.dropna(subset=["Estimation_point"])
    traffic = traffic.dropna(subset=["Estimation_point"])

    clusters["Estimation_point"] = clusters["Estimation_point"].astype(int)
    traffic["Estimation_point"] = traffic["Estimation_point"].astype(int)

    # --- Merge ---
    if "AAWDT" not in traffic.columns:
        raise ValueError("Your traffic CSV must contain an 'AAWDT' column.")

    merged = pd.merge(
        clusters[["Estimation_point", "filename", "cluster"]],
        traffic,
        on="Estimation_point",
        how="inner"
    )

    if merged.empty:
        raise RuntimeError(
            "Merged dataframe is empty. Check that filename IDs match Estimation_point values."
        )

    # Save merged dataframe
    merged.to_csv(OUT_MERGED_CSV, index=False)

    # --- Per-cluster statistics for AAWDT ---
    stats_tbl = (
        merged.groupby("cluster")["AAWDT"]
        .agg(count="count", mean="mean", median="median", std="std", min="min", max="max")
        .sort_index()
    )

    # --- ANOVA + Kruskal-Wallis ---
    groups = [g["AAWDT"].values for _, g in merged.groupby("cluster")]
    anova_res = f_oneway(*groups)
    kruskal_res = kruskal(*groups)

    # --- Eta-squared effect size ---
    grand_mean = merged["AAWDT"].mean()
    ss_between = sum(
        len(g) * (g["AAWDT"].mean() - grand_mean) ** 2
        for _, g in merged.groupby("cluster")
    )
    ss_total = ((merged["AAWDT"] - grand_mean) ** 2).sum()
    eta_sq = float(ss_between / ss_total)

    # --- Spearman correlation (rank clusters by median AAWDT) ---
    # Create an ordered rank based on the cluster medians
    medians = stats_tbl["median"].sort_values()  # low -> high
    rank_map = {cluster_id: rank+1 for rank, cluster_id in enumerate(medians.index.tolist())}

    merged["cluster_rank"] = merged["cluster"].map(rank_map)

    rho, rho_p = spearmanr(merged["cluster_rank"], merged["AAWDT"])

    # --- Save report ---
    with open(OUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("AAWDT vs Image Clusters (KMeans)\n")
        f.write("================================\n\n")

        f.write("Per-cluster AAWDT stats:\n")
        f.write(stats_tbl.to_string(float_format=lambda x: f"{x:,.3f}"))
        f.write("\n\n")

        f.write("Cluster rank mapping (by median AAWDT, low→high):\n")
        for k, v in sorted(rank_map.items(), key=lambda kv: kv[1]):
            f.write(f"  Cluster {k} -> Rank {v}\n")
        f.write("\n")

        f.write(f"ANOVA p-value: {anova_res.pvalue:.6e}\n")
        f.write(f"Kruskal-Wallis p-value: {kruskal_res.pvalue:.6e}\n")
        f.write(f"Effect size (eta squared): {eta_sq:.4f}\n\n")

        f.write("Spearman correlation (cluster_rank vs AAWDT):\n")
        f.write(f"  rho: {rho:.4f}\n")
        f.write(f"  p-value: {rho_p:.6e}\n")

    # --- Plots ---
    sns.set_context("talk")

    # Boxplot: AAWDT by cluster
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=merged, x="cluster", y="AAWDT")
    plt.title("AAWDT Distribution by Image Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("AAWDT")
    plt.tight_layout()
    plt.savefig(OUT_BOXPLOT, dpi=200)
    plt.close()

    # Scatter: rank vs AAWDT (jitter for visibility)
    plt.figure(figsize=(10, 6))
    x = merged["cluster_rank"].astype(float)
    jitter = np.random.uniform(-0.08, 0.08, size=len(x))
    plt.scatter(x + jitter, merged["AAWDT"], alpha=0.35)
    plt.title(f"AAWDT vs Cluster Rank (Spearman rho={rho:.2f})")
    plt.xlabel("Cluster Rank (low → high by median AAWDT)")
    plt.ylabel("AAWDT")
    plt.xticks(sorted(merged["cluster_rank"].unique()))
    plt.tight_layout()
    plt.savefig(OUT_SCATTER, dpi=200)
    plt.close()

    # --- Console summary ---
    print("\n✅ Saved merged dataframe:", OUT_MERGED_CSV)
    print("✅ Saved report:", OUT_REPORT_TXT)
    print("✅ Saved plots:", OUT_BOXPLOT, "and", OUT_SCATTER)

    print("\n--- Quick Results ---")
    print(stats_tbl)
    print("\nRank map (low→high):", rank_map)
    print(f"\nSpearman rho={rho:.4f}, p={rho_p:.3e}")
    print(f"Eta^2={eta_sq:.4f}")
    print(f"ANOVA p={anova_res.pvalue:.3e}, Kruskal p={kruskal_res.pvalue:.3e}")


if __name__ == "__main__":
    main()