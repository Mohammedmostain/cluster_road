import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

WORK_DIR = Path("work")

FN_PATH = WORK_DIR / "filenames.txt"
TRAFFIC_CSV = Path("traffic_data.csv")

OUT_META_NPY = WORK_DIR / "metadata_features.npy"
OUT_MASK_NPY = WORK_DIR / "metadata_match_mask.npy"
OUT_INFO_JSON = WORK_DIR / "metadata_info.json"

META_COLS = ["Speed", "Lanes"]

def extract_id_from_filename(fn: str):
    """
    Extract the last integer appearing in the filename (without extension).
    Examples:
      '123.png' -> 123
      'road_00123_zoom.png' -> 123
      'abc.png' -> None
    """
    stem = Path(fn).stem
    nums = re.findall(r"\d+", stem)
    if not nums:
        return None
    return int(nums[-1])

def main():
    # --- load filenames in the SAME order as embeddings ---
    fns = [x.strip() for x in open(FN_PATH, "r", encoding="utf-8") if x.strip()]
    if len(fns) == 0:
        raise ValueError(f"{FN_PATH} is empty or missing.")

    # --- load traffic ---
    df = pd.read_csv(TRAFFIC_CSV)

    if "Estimation_point" not in df.columns:
        raise ValueError("traffic_data.csv missing required column: Estimation_point")

    # Normalize join key
    df["Estimation_point"] = pd.to_numeric(df["Estimation_point"], errors="coerce").astype("Int64")

    # Build filename -> Estimation_point
    fn_ids = [extract_id_from_filename(fn) for fn in fns]
    clusters_df = pd.DataFrame({"filename": fns, "Estimation_point": pd.array(fn_ids, dtype="Int64")})

    # Diagnostics: filename parsing
    n_parsed = clusters_df["Estimation_point"].notna().sum()
    print(f"Filenames total: {len(fns)}")
    print(f"Parsed numeric IDs from filenames: {n_parsed} / {len(fns)}")

    # LEFT merge to keep alignment and see misses
    merged = pd.merge(clusters_df, df, on="Estimation_point", how="left", indicator=True)

    # Diagnostics: merge success
    n_matched = (merged["_merge"] == "both").sum()
    n_left_only = (merged["_merge"] == "left_only").sum()
    print(f"Matched traffic rows: {n_matched} / {len(fns)}")
    print(f"No match in traffic_data.csv: {n_left_only} / {len(fns)}")

    # Check metadata cols exist
    missing_cols = [c for c in META_COLS if c not in merged.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in traffic_data.csv: {missing_cols}")

    # Extract metadata in the same order as filenames.txt
    X_meta = merged[META_COLS].copy()
    for c in META_COLS:
        X_meta[c] = pd.to_numeric(X_meta[c], errors="coerce")

    # Mask for rows that actually matched on Estimation_point
    match_mask = (merged["_merge"] == "both").to_numpy()

    # Impute missing values so we keep row alignment
    imputer = SimpleImputer(strategy="median")
    X_meta_imp = imputer.fit_transform(X_meta).astype(np.float32)

    # Scale
    scaler = StandardScaler()
    X_meta_scaled = scaler.fit_transform(X_meta_imp).astype(np.float32)

    # Save aligned outputs
    np.save(OUT_META_NPY, X_meta_scaled)
    np.save(OUT_MASK_NPY, match_mask)

    info = {
        "meta_cols": META_COLS,
        "n_filenames": int(len(fns)),
        "n_parsed_ids": int(n_parsed),
        "n_matched": int(n_matched),
        "n_unmatched": int(n_left_only),
        "imputer": "median",
        "scaler": "StandardScaler",
        "note": "metadata_features.npy is aligned 1:1 with filenames.txt; unmatched rows were imputed. metadata_match_mask.npy shows which rows had a traffic match."
    }
    OUT_INFO_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_INFO_JSON, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("Saved:", OUT_META_NPY)
    print("Saved:", OUT_MASK_NPY)
    print("Saved:", OUT_INFO_JSON)
    print("X_meta shape:", X_meta_scaled.shape)

if __name__ == "__main__":
    main()
