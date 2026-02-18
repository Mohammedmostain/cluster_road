import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, f_oneway, kruskal
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# =========================
# CONFIG
# =========================
WORK_DIR = Path("work")

# Pick ONE at a time (or run twice with different values)
METHOD = "multimodal"   # "kmeans" or "hdbscan" or "dbscan" or "multimodal"

CLUSTER_LABELS_PATH = {
    "kmeans":     WORK_DIR / "kmeans_labels.csv",
    "hdbscan":    WORK_DIR / "hdbscan_labels.csv",
    "dbscan":     WORK_DIR / "dbscan_labels.csv",
    "multimodal": WORK_DIR / "multimodal_hdbscan" / "labels.csv",  # <-- default
}[METHOD]

TRAFFIC_CSV_PATH = Path("traffic_data.csv")

# Remove noise cluster -1 (common in DBSCAN/HDBSCAN)
REMOVE_NOISE = METHOD in {"hdbscan", "dbscan", "multimodal"}

# If you want Road Class similarity metrics
COMPUTE_ROADCLASS_ALIGNMENT = True
ROAD_CLASS_COL = "Road Class"  # adjust if your column name differs

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
OUT_CROSSTAB_CSV = OUT_DIR / f"crosstab_cluster_vs_roadclass_{METHOD}.csv"
OUT_ALIGNMENT_JSON = OUT_DIR / f"alignment_cluster_vs_roadclass_{METHOD}.json"


# =========================
# HELPERS
# =========================
def extract_id_from_filename(fn: str):
    """
    Extract the last integer appearing in the filename stem.
    Examples:
      '123.png' -> 123
      'road_00123_zoom.png' -> 123
      'abc.png' -> None
    """
    stem = Path(str(fn)).stem
    nums = re.findall(r"\d+", stem)
    if not nums:
        return None
    return int(nums[-1])


def main():
    # --- Load files ---
    clusters = pd.read_csv(CLUSTER_LABELS_PATH)
    traffic = pd.read_csv(TRAFFIC_CSV_PATH)

    # Validate expected columns early
    required_cluster_cols = {"filename", "cluster"}
    missing = required_cluster_cols - set(clusters.columns)
    if missing:
        raise ValueError(
            f"Cluster labels file is missing columns: {missing}. "
            f"Expected at least {required_cluster_cols}. "
            f"Got columns: {list(clusters.columns)}"
        )

    if "Estimation_point" not in traffic.columns:
        raise ValueError("Traffic CSV must contain an 'Estimation_point' column.")
    if "AAWDT" not in traffic.columns:
        raise ValueError("Traffic CSV must contain an 'AAWDT' column.")

    # --- Normalize keys ---
    # Use robust regex extraction instead of assuming filename is numeric
    clusters["Estimation_point"] = clusters["filename"].apply(extract_id_from_filename)
    clusters["Estimation_point"] = pd.to_numeric(clusters["Estimation_point"], errors="coerce")
    traffic["Estimation_point"] = pd.to_numeric(traffic["Estimation_point"], errors="coerce")

    clusters = clusters.dropna(subset=["Estimation_point"]).copy()
    traffic = traffic.dropna(subset=["Estimation_point"]).copy()

    clusters["Estimation_point"] = clusters["Estimation_point"].astype(int)
    traffic["Estimation_point"] = traffic["Estimation_point"].astype(int)

    # --- Merge ---
    merged = pd.merge(
        clusters[["Estimation_point", "filename", "cluster"]],
        traffic,
        on="Estimation_point",
        how="inner"
    )

    if REMOVE_NOISE:
        merged = merged[merged["cluster"] != -1].copy()

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

    # --- Road Class alignment (ARI/NMI + crosstab) ---
    alignment = None
    if COMPUTE_ROADCLASS_ALIGNMENT:
        if ROAD_CLASS_COL in merged.columns:
            # ARI/NMI expects label arrays
            ari = adjusted_rand_score(merged[ROAD_CLASS_COL].astype(str), merged["cluster"].astype(int))
            nmi = normalized_mutual_info_score(merged[ROAD_CLASS_COL].astype(str), merged["cluster"].astype(int))

            ctab = pd.crosstab(merged["cluster"], merged[ROAD_CLASS_COL])
            ctab.to_csv(OUT_CROSSTAB_CSV)

            alignment = {"ARI": float(ari), "NMI": float(nmi)}
            with open(OUT_ALIGNMENT_JSON, "w", encoding="utf-8") as f:
                json.dump(alignment, f, indent=2)
        else:
            alignment = {"error": f"Column '{ROAD_CLASS_COL}' not found in merged traffic data."}
            with open(OUT_ALIGNMENT_JSON, "w", encoding="utf-8") as f:
                json.dump(alignment, f, indent=2)

    # --- Report ---
    with open(OUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(f"AAWDT vs Clusters ({METHOD.upper()})\n")
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
        f.write(f"  p-value: {rho_p:.6e}\n\n")

        if alignment is not None:
            f.write("Alignment with Road Class:\n")
            f.write(json.dumps(alignment, indent=2))
            f.write("\n")
            if (OUT_CROSSTAB_CSV.exists()):
                f.write(f"\nCrosstab saved: {OUT_CROSSTAB_CSV}\n")

    # --- Plots ---
    sns.set_context("talk")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=merged, x="cluster", y="AAWDT")
    plt.title(f"AAWDT Distribution by Cluster ({METHOD.upper()})")
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

    if alignment is not None and "ARI" in alignment:
        print(f"\nAlignment vs Road Class: ARI={alignment['ARI']:.4f}, NMI={alignment['NMI']:.4f}")
        print("Crosstab:", OUT_CROSSTAB_CSV)


if __name__ == "__main__":
    main()
