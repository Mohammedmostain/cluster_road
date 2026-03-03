from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Load:
# cluster_labels.csv (filename, cluster)
# traffic_data.csv (Estimation_point, Road Class)

clusters = pd.read_csv(ROOT / "work" / "hdbscan_labels.csv")
traffic = pd.read_csv(ROOT / "data" / "traffic_data.csv")

clusters["Estimation_point"] = (
    clusters["filename"]
    .str.replace(".png", "", regex=False)
    .astype(int)
)

merged = pd.merge(
    clusters,
    traffic[["Estimation_point", "Road Class"]],
    on="Estimation_point",
    how="inner"
)

# Remove noise if HDBSCAN
merged = merged[merged["cluster"] != -1]

ari = adjusted_rand_score(
    merged["Road Class"],
    merged["cluster"]
)

nmi = normalized_mutual_info_score(
    merged["Road Class"],
    merged["cluster"]
)

print("ARI:", ari)
print("NMI:", nmi)
