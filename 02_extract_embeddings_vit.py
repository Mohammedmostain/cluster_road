import json
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm
from timm.models.vision_transformer import resize_pos_embed


# =========================
# CONFIG
# =========================
MANIFEST_PATH = Path("work/pairs_manifest.csv")
OUT_DIR = Path("work")

BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True

VIT_NAME = "vit_base_patch14_dinov2"
IMG_SIZE = 224

FUSION = "concat"  # "concat" or "avg"

REMOVE_REDLINE = True
RED_HSV1 = ((0, 70, 50), (10, 255, 255))
RED_HSV2 = ((170, 70, 50), (180, 255, 255))
INPAINT_RADIUS = 3

EMB_PATH = OUT_DIR / "embeddings.npy"
FN_PATH = OUT_DIR / "filenames.txt"
META_PATH = OUT_DIR / "embeddings_meta.json"


# =========================
# RED LINE REMOVAL
# =========================
def red_mask_hsv(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    (l1, u1) = RED_HSV1
    (l2, u2) = RED_HSV2
    mask1 = cv2.inRange(hsv, np.array(l1, dtype=np.uint8), np.array(u1, dtype=np.uint8))
    mask2 = cv2.inRange(hsv, np.array(l2, dtype=np.uint8), np.array(u2, dtype=np.uint8))
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    return mask


def remove_redline_inpaint(img_bgr: np.ndarray) -> np.ndarray:
    mask = red_mask_hsv(img_bgr)
    if mask.max() == 0:
        return img_bgr
    return cv2.inpaint(img_bgr, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)


# =========================
# DATASET
# =========================
class PairedRoadDataset(Dataset):
    def __init__(self, manifest_csv: Path, tfm):
        self.items = []
        self.tfm = tfm

        with open(manifest_csv, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            idx_fn = header.index("filename")
            idx_in = header.index("zoom_in_path")
            idx_out = header.index("zoom_out_path")

            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                fn = parts[idx_fn]
                p_in = parts[idx_in]
                p_out = parts[idx_out]
                self.items.append((fn, p_in, p_out))

        if not self.items:
            raise RuntimeError(f"No pairs found in manifest: {manifest_csv}")

    def __len__(self):
        return len(self.items)

    def _read_image(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")

        if REMOVE_REDLINE:
            img = remove_redline_inpaint(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        fn, p_in, p_out = self.items[idx]
        img_in = self._read_image(p_in)
        img_out = self._read_image(p_out)
        x_in = self.tfm(img_in)
        x_out = self.tfm(img_out)
        return fn, x_in, x_out


def build_transform(img_size: int):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


# =========================
# MODEL FIX: resize pos_embed for IMG_SIZE (YOUR TIMM SIGNATURE)
# =========================
def adapt_vit_to_img_size(model, img_size: int):
    if not hasattr(model, "pos_embed") or model.pos_embed is None:
        return model

    # Patch size
    patch_size = model.patch_embed.patch_size
    patch = patch_size[0] if isinstance(patch_size, tuple) else int(patch_size)

    # New grid size in patch units
    gs_new = (img_size // patch, img_size // patch)

    # Determine number of prefix tokens (cls token only for ViT)
    num_prefix = 1

    # Create a dummy new pos_embed tensor with correct token count
    # tokens = prefix + (H/patch)*(W/patch)
    num_tokens_new = num_prefix + gs_new[0] * gs_new[1]
    posemb_new = torch.zeros((1, num_tokens_new, model.pos_embed.shape[-1]),
                             device=model.pos_embed.device, dtype=model.pos_embed.dtype)

    # Your timm signature:
    # resize_pos_embed(posemb, posemb_new, num_prefix_tokens=1, gs_new=(h,w), ...)
    new_pos = resize_pos_embed(model.pos_embed, posemb_new, num_prefix, gs_new)

    model.pos_embed = torch.nn.Parameter(new_pos)

    # update metadata to avoid asserts
    if hasattr(model.patch_embed, "img_size"):
        model.patch_embed.img_size = (img_size, img_size)
    if hasattr(model, "img_size"):
        model.img_size = (img_size, img_size)

    return model


# =========================
# FEATURE EXTRACTION
# =========================
@torch.no_grad()
def extract_features(model, x, device):
    x = x.to(device, non_blocking=True)
    with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
        feats = model(x)
    if feats.ndim != 2:
        feats = feats.view(feats.size(0), -1)
    return feats


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print("CUDA OK:", torch.cuda.get_device_name(0))

    tfm = build_transform(IMG_SIZE)
    ds = PairedRoadDataset(MANIFEST_PATH, tfm)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = timm.create_model(VIT_NAME, pretrained=True, num_classes=0)
    model = adapt_vit_to_img_size(model, IMG_SIZE)
    model.eval().to(device)

    all_emb = []
    all_fns = []

    for fns, x_in, x_out in tqdm(dl, desc="Embedding"):
        e_in = extract_features(model, x_in, device)
        e_out = extract_features(model, x_out, device)

        if FUSION == "concat":
            e = torch.cat([e_in, e_out], dim=1)
        elif FUSION == "avg":
            e = (e_in + e_out) * 0.5
        else:
            raise ValueError(f"Unknown FUSION: {FUSION}")

        all_emb.append(e.cpu().numpy())
        all_fns.extend(list(fns))

    emb = np.concatenate(all_emb, axis=0)
    print("Embeddings shape:", emb.shape)

    np.save(EMB_PATH, emb)
    with open(FN_PATH, "w", encoding="utf-8") as f:
        for fn in all_fns:
            f.write(fn + "\n")

    meta = {
        "vit_name": VIT_NAME,
        "img_size": IMG_SIZE,
        "fusion": FUSION,
        "remove_redline": REMOVE_REDLINE,
        "inpaint_radius": INPAINT_RADIUS,
        "num_items": len(all_fns),
        "embedding_dim": int(emb.shape[1]),
        "manifest": str(MANIFEST_PATH),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {EMB_PATH}")
    print(f"Saved: {FN_PATH}")
    print(f"Saved: {META_PATH}")


if __name__ == "__main__":
    main()
