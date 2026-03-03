import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, f_oneway, kruskal
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent

# =========================
# CONFIG
# =========================
TRAFFIC_CSV_PATH = ROOT / "data" / "traffic_data.csv"
OUT_DIR = ROOT / "work" / "road_class_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_REPORT_TXT = OUT_DIR / "road_class_report.txt"
OUT_BOXPLOT = OUT_DIR / "aawdt_by_road_class.png"
OUT_SCATTER = OUT_DIR / "aawdt_vs_road_class_rank.png"


def main():

    # --- Load data ---
    df = pd.read_csv(TRAFFIC_CSV_PATH)

    if "Road Class" not in df.columns:
        raise ValueError("Column 'Road Class' not found in traffic CSV.")

    if "AAWDT" not in df.columns:
        raise ValueError("Column 'AAWDT' not found in traffic CSV.")

    # Drop missing values
    df = df.dropna(subset=["Road Class", "AAWDT"])

    # --- Per-class statistics ---
    stats_tbl = (
        df.groupby("Road Class")["AAWDT"]
        .agg(count="count", mean="mean", median="median", std="std", min="min", max="max")
        .sort_values("median")
    )

    # --- ANOVA + Kruskal ---
    groups = [g["AAWDT"].values for _, g in df.groupby("Road Class")]
    anova_res = f_oneway(*groups)
    kruskal_res = kruskal(*groups)

    # --- Eta-squared ---
    grand_mean = df["AAWDT"].mean()
    ss_between = sum(
        len(g) * (g["AAWDT"].mean() - grand_mean) ** 2
        for _, g in df.groupby("Road Class")
    )
    ss_total = ((df["AAWDT"] - grand_mean) ** 2).sum()
    eta_sq = float(ss_between / ss_total)

    # --- Spearman (rank by median AAWDT) ---
    rank_map = {
        road_class: rank + 1
        for rank, road_class in enumerate(stats_tbl.index.tolist())
    }

    df["road_class_rank"] = df["Road Class"].map(rank_map)

    rho, rho_p = spearmanr(df["road_class_rank"], df["AAWDT"])

    # --- Save report ---
    with open(OUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("AAWDT vs Road Class Analysis\n")
        f.write("============================\n\n")

        f.write("Per-road-class AAWDT stats:\n")
        f.write(stats_tbl.to_string(float_format=lambda x: f"{x:,.3f}"))
        f.write("\n\n")

        f.write("Road class rank mapping (low→high median AAWDT):\n")
        for k, v in sorted(rank_map.items(), key=lambda kv: kv[1]):
            f.write(f"  {k} -> Rank {v}\n")
        f.write("\n")

        f.write(f"ANOVA p-value: {anova_res.pvalue:.6e}\n")
        f.write(f"Kruskal-Wallis p-value: {kruskal_res.pvalue:.6e}\n")
        f.write(f"Effect size (eta squared): {eta_sq:.4f}\n\n")

        f.write("Spearman correlation (road_class_rank vs AAWDT):\n")
        f.write(f"  rho: {rho:.4f}\n")
        f.write(f"  p-value: {rho_p:.6e}\n")

    # --- Plots ---
    sns.set_context("talk")

    # Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Road Class", y="AAWDT")
    plt.xticks(rotation=45)
    plt.title("AAWDT Distribution by Road Class")
    plt.tight_layout()
    plt.savefig(OUT_BOXPLOT, dpi=200)
    plt.close()

    # Scatter
    plt.figure(figsize=(10, 6))
    jitter = np.random.uniform(-0.08, 0.08, size=len(df))
    plt.scatter(df["road_class_rank"] + jitter, df["AAWDT"], alpha=0.35)
    plt.xticks(sorted(df["road_class_rank"].unique()))
    plt.xlabel("Road Class Rank (low → high median AAWDT)")
    plt.ylabel("AAWDT")
    plt.title(f"AAWDT vs Road Class Rank (Spearman rho={rho:.2f})")
    plt.tight_layout()
    plt.savefig(OUT_SCATTER, dpi=200)
    plt.close()

    # --- Console summary ---
    print("\n--- Road Class Results ---")
    print(stats_tbl)
    print("\nRank map (low→high):", rank_map)
    print(f"\nSpearman rho={rho:.4f}, p={rho_p:.3e}")
    print(f"Eta^2={eta_sq:.4f}")
    print(f"ANOVA p={anova_res.pvalue:.3e}, Kruskal p={kruskal_res.pvalue:.3e}")


if __name__ == "__main__":
    main()
