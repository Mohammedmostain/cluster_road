import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ===== CONFIG =====
WORK_DIR = ROOT / "work"

CLUSTER_LABELS = WORK_DIR / "kmeans_labels.csv"

# Replace with your actual CSV path
TRAFFIC_CSV = ROOT / "data" / "traffic_data.csv"

OUTPUT_REPORT = WORK_DIR / "aawdt_cluster_report.txt"
BOXPLOT_OUT = WORK_DIR / "aawdt_by_cluster.png"


def main():

    print("Loading data...")

    clusters = pd.read_csv(CLUSTER_LABELS)
    traffic = pd.read_csv(TRAFFIC_CSV)

    # ---- Prepare join key ----
    # Your CSV uses "Estimation_point" as image id
    # But cluster file has filename like "1182163.png"

    clusters["Estimation_point"] = clusters["filename"].str.replace(".png", "", regex=False).astype(int)

    merged = pd.merge(
        clusters,
        traffic[["Estimation_point", "AAWDT"]],
        on="Estimation_point",
        how="inner"
    )

    print(f"Merged rows: {len(merged)}")

    # ===== BASIC STATISTICS =====
    summary = merged.groupby("cluster")["AAWDT"].agg([
        "count", "mean", "median", "std", "min", "max"
    ])

    print("\nPer-cluster AAWDT statistics:")
    print(summary)

    # ===== ANOVA TEST =====
    cluster_groups = [
        group["AAWDT"].values
        for _, group in merged.groupby("cluster")
    ]

    anova = stats.f_oneway(*cluster_groups)

    # ===== Kruskal-Wallis (non-parametric) =====
    kruskal = stats.kruskal(*cluster_groups)

    # ===== Effect size (eta squared) =====
    grand_mean = merged["AAWDT"].mean()

    ss_between = sum(
        len(g) * (g["AAWDT"].mean() - grand_mean) ** 2
        for _, g in merged.groupby("cluster")
    )

    ss_total = sum((merged["AAWDT"] - grand_mean) ** 2)

    eta_squared = ss_between / ss_total


    # ===== Save report =====
    with open(OUTPUT_REPORT, "w") as f:
        f.write("AAWDT vs Cluster Analysis\n")
        f.write("========================\n\n")

        f.write("Per-cluster statistics:\n")
        f.write(summary.to_string())
        f.write("\n\n")

        f.write(f"ANOVA p-value: {anova.pvalue}\n")
        f.write(f"Kruskal-Wallis p-value: {kruskal.pvalue}\n")
        f.write(f"Effect size (eta squared): {eta_squared:.4f}\n")

    print("\nSaved report to:", OUTPUT_REPORT)


    # ===== Visualization =====
    plt.figure(figsize=(10, 6))

    sns.boxplot(data=merged, x="cluster", y="AAWDT")
    plt.title("AAWDT Distribution by Image Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("AAWDT")

    plt.tight_layout()
    plt.savefig(BOXPLOT_OUT)
    plt.close()

    print("Saved boxplot to:", BOXPLOT_OUT)


if __name__ == "__main__":
    main()
