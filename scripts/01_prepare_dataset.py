import os
import cv2
import csv
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# =========================
# CONFIG
# =========================
ZOOM_IN_DIR  = ROOT / "data" / "zoom_in"
ZOOM_OUT_DIR = ROOT / "data" / "zoom_out"

OUT_DIR = ROOT / "work"              # all intermediate outputs go here
MANIFEST_PATH = OUT_DIR / "pairs_manifest.csv"
PREVIEW_DIR = OUT_DIR / "previews_redmask"

REMOVE_REDLINE = True               # set False to keep as-is
RED_HSV1 = ((0, 70, 50), (10, 255, 255))     # lower red range in HSV
RED_HSV2 = ((170, 70, 50), (180, 255, 255))  # upper red range in HSV
INPAINT_RADIUS = 3                  # 3-7 typical

PREVIEW_COUNT = 16                  # number of random pairs to preview

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# =========================
# UTILS
# =========================
def list_images(d: Path) -> List[str]:
    return sorted([p.name for p in d.iterdir() if p.suffix.lower() in IMG_EXTS])

def find_pairs(dir_a: Path, dir_b: Path) -> List[Tuple[Path, Path, str]]:
    a = set(list_images(dir_a))
    b = set(list_images(dir_b))
    common = sorted(a.intersection(b))
    pairs = [(dir_a / fn, dir_b / fn, fn) for fn in common]
    missing_a = sorted(list(b - a))
    missing_b = sorted(list(a - b))
    return pairs, missing_a, missing_b

def red_mask_hsv(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    (l1, u1) = RED_HSV1
    (l2, u2) = RED_HSV2
    mask1 = cv2.inRange(hsv, np.array(l1, dtype=np.uint8), np.array(u1, dtype=np.uint8))
    mask2 = cv2.inRange(hsv, np.array(l2, dtype=np.uint8), np.array(u2, dtype=np.uint8))
    mask = cv2.bitwise_or(mask1, mask2)

    # Clean up small speckles and connect thin line segments
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    return mask

def remove_redline_inpaint(img_bgr: np.ndarray) -> np.ndarray:
    mask = red_mask_hsv(img_bgr)
    if mask.max() == 0:
        return img_bgr
    # Inpaint replaces masked pixels using surrounding texture
    out = cv2.inpaint(img_bgr, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    return out

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# =========================
# MAIN
# =========================
def main():
    ensure_dir(OUT_DIR)
    ensure_dir(PREVIEW_DIR)

    pairs, missing_in, missing_out = find_pairs(ZOOM_IN_DIR, ZOOM_OUT_DIR)

    print(f"Found pairs: {len(pairs)}")
    if missing_in:
        print(f"WARNING: {len(missing_in)} files exist in zoom_out but not zoom_in (showing up to 10): {missing_in[:10]}")
    if missing_out:
        print(f"WARNING: {len(missing_out)} files exist in zoom_in but not zoom_out (showing up to 10): {missing_out[:10]}")

    # Write manifest
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "zoom_in_path", "zoom_out_path"])
        for p_in, p_out, fn in pairs:
            w.writerow([fn, str(p_in), str(p_out)])

    print(f"Saved manifest -> {MANIFEST_PATH}")

    # Previews
    preview_pairs = random.sample(pairs, k=min(PREVIEW_COUNT, len(pairs)))
    for p_in, p_out, fn in preview_pairs:
        img_in = cv2.imread(str(p_in), cv2.IMREAD_COLOR)
        img_out = cv2.imread(str(p_out), cv2.IMREAD_COLOR)
        if img_in is None or img_out is None:
            continue

        if REMOVE_REDLINE:
            img_in_clean = remove_redline_inpaint(img_in)
            img_out_clean = remove_redline_inpaint(img_out)
        else:
            img_in_clean, img_out_clean = img_in, img_out

        # save side-by-side comparisons for quick inspection
        def side_by_side(a, b):
            h = max(a.shape[0], b.shape[0])
            # pad to same height
            def pad_to(img, hh):
                if img.shape[0] == hh:
                    return img
                pad = hh - img.shape[0]
                return cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            a2, b2 = pad_to(a, h), pad_to(b, h)
            return np.concatenate([a2, b2], axis=1)

        cv2.imwrite(str(PREVIEW_DIR / f"{fn}_in_before_after.png"),  side_by_side(img_in, img_in_clean))
        cv2.imwrite(str(PREVIEW_DIR / f"{fn}_out_before_after.png"), side_by_side(img_out, img_out_clean))

    print(f"Saved previews -> {PREVIEW_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
