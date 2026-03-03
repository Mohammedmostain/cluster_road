# Unsupervised Road Image Clustering with Vision Transformers

## Overview

This project performs **unsupervised clustering of satellite road images** using a multi-view Vision Transformer (ViT) pipeline. The goal is to automatically group road images into meaningful categories **without any ground-truth labels**, and analyze how the discovered clusters relate to real-world traffic metrics such as **AAWDT (Annual Average Weekday Traffic).**

The system processes paired satellite images captured at two zoom levels, learns visual representations using **self-supervised ViT embeddings (DINOv2)**, and clusters them using either **K-means with automatic elbow-based cluster selection** or **HDBSCAN with optional multimodal fusion**.

---

## Project Goals

* Learn representations of road images without supervision
* Use paired zoom-in / zoom-out images jointly
* Discover natural groupings of road types
* Select optimal cluster count automatically
* Export clustered images for qualitative analysis
* Analyze correlation between image clusters and traffic data (AAWDT)

---

## Pipeline Summary

### 1. Data Preparation

* Two directories of satellite images (~750 each)
* Same filenames represent the same road segment at two zoom levels
* Optional removal of red road-line overlay via inpainting
* Automatic pairing using filenames

### 2. Representation Learning

* Pretrained **DINOv2 Vision Transformer** used as feature extractor
* GPU-accelerated embedding extraction
* Multi-view fusion of zoom levels (concatenation)
* Outputs one embedding per road segment

### 3. Clustering (two options)

* Standardization + PCA dimensionality reduction
* **K-means** with elbow method for automatic K selection, or
* **HDBSCAN** with UMAP preprocessing (image-only or multimodal)
* Cluster assignments saved and visualized

### 4. Evaluation

* Export clustered images into folders
* Statistical analysis linking clusters to AAWDT and road class
* ANOVA, Kruskal-Wallis, Spearman, ARI, NMI, and effect size calculations

---

## Repository Structure

```
cluster_road/
│
├── data/
│   ├── zoom_in/            # Satellite images, close zoom
│   ├── zoom_out/           # Satellite images, wide zoom
│   └── traffic_data.csv    # AAWDT and road class ground truth
│
├── work/                   # All generated outputs (gitignored)
│   ├── pairs_manifest.csv
│   ├── embeddings.npy
│   ├── filenames.txt
│   ├── kmeans_labels.csv
│   ├── kmeans_elbow.png
│   ├── clusters_kmeans/
│   ├── multimodal_hdbscan/
│   └── ...
│
├── scripts/
│   ├── 01_prepare_dataset.py       # Pair images, remove red overlays
│   ├── 02_extract_embeddings_vit.py# DINOv2 embedding extraction
│   ├── 03a_kmeans_cluster.py       # K-Means + elbow clustering
│   ├── 03b_hdbscan_cluster.py      # HDBSCAN clustering (image only)
│   ├── 04_aawdt_analysis.py        # ANOVA: clusters vs AAWDT
│   ├── 05_spearman_analysis.py     # Spearman, ARI, NMI analysis
│   ├── 06_compare_road_class.py    # Compare clusters vs road class
│   ├── 07_road_class_aawdt.py      # Road class AAWDT baseline
│   ├── 08_k_sweep.py               # Grid search over K values
│   ├── 09_build_metadata.py        # Build metadata feature matrix
│   ├── 10_hdbscan_multimodal.py    # HDBSCAN with image + metadata
│   ├── 11_viz_umap.py              # 2D UMAP visualization
│   └── random_grid.py              # Utility: random 2x2 image grid
│
├── requirements.txt
└── README.md
```

---

## Requirements

* Python 3.10+
* CUDA-enabled GPU recommended

### Python Packages

```
torch
torchvision
timm
hdbscan
umap-learn
opencv-python
numpy
pandas
scikit-learn
matplotlib
seaborn
scipy
Pillow
tqdm
```

Install dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm hdbscan umap-learn opencv-python numpy pandas scikit-learn matplotlib seaborn scipy Pillow tqdm
```

---

## Usage

All scripts are in the `scripts/` directory and can be run from the project root. They resolve all paths relative to the project root automatically.

### Step 1 – Prepare Dataset

Pairs zoom-in and zoom-out images, optionally removes red road-line overlays:

```bash
python scripts/01_prepare_dataset.py
```

Output: `work/pairs_manifest.csv`

---

### Step 2 – Extract ViT Embeddings

Runs DINOv2 over each image pair and concatenates embeddings:

```bash
python scripts/02_extract_embeddings_vit.py
```

Outputs:
* `work/embeddings.npy`
* `work/filenames.txt`

---

### Step 3 – Cluster

Choose one (or both):

#### Option A — K-Means with Elbow

```bash
python scripts/03a_kmeans_cluster.py
```

Outputs:
* `work/kmeans_elbow.png`
* `work/kmeans_labels.csv`
* `work/clusters_kmeans/cluster_X/{zoom_in,zoom_out}/`

#### Option B — HDBSCAN (image embeddings only)

```bash
python scripts/03b_hdbscan_cluster.py
```

Output: `work/hdbscan_labels.csv`

#### Option C — HDBSCAN with Metadata (multimodal)

First build the metadata feature matrix, then cluster:

```bash
python scripts/09_build_metadata.py
python scripts/10_hdbscan_multimodal.py
```

Outputs: `work/multimodal_hdbscan/labels.csv`, `work/multimodal_hdbscan/metrics.json`

---

### Step 4 – Analyze Results

#### AAWDT vs clusters (basic)

```bash
python scripts/04_aawdt_analysis.py
```

#### Full analysis — Spearman, ARI, NMI

```bash
python scripts/05_spearman_analysis.py
```

#### Compare clusters to road class labels

```bash
python scripts/06_compare_road_class.py
```

#### Road class AAWDT baseline

```bash
python scripts/07_road_class_aawdt.py
```

---

### Optional: Visualization and Utilities

#### UMAP 2D visualization of HDBSCAN clusters

```bash
python scripts/11_viz_umap.py
```

Output: `work/viz_hdbscan_umap2d.png`

#### K sweep (grid search over K values)

```bash
python scripts/08_k_sweep.py
```

#### Random image grid from a cluster folder

```bash
python scripts/random_grid.py
```

---

## Key Methods

### Representation Learning

* Self-supervised **DINOv2 ViT** (`vit_base_patch14_dinov2`)
* No labels required
* Multi-view fusion: zoom_in + zoom_out embeddings concatenated (1536-dim)

### Clustering Strategy

| Script | Method | Preprocessing |
|---|---|---|
| `03a_kmeans_cluster.py` | K-Means | StandardScaler + PCA (95% var) |
| `03b_hdbscan_cluster.py` | HDBSCAN | StandardScaler + PCA + UMAP |
| `10_hdbscan_multimodal.py` | HDBSCAN | Image + metadata fusion, PCA 64-dim + UMAP 30-dim |

### Statistical Validation

* Spearman rank correlation (clusters vs AAWDT)
* ANOVA and Kruskal-Wallis tests
* Effect size (eta-squared)
* Adjusted Rand Index and Normalized Mutual Information (vs road class)

---

## Results

The pipeline produces:

* Meaningful clusters of road imagery
* Visual grouping of similar road types
* Quantitative insights into how clusters relate to traffic volume and road class
* Fully reproducible workflow from images to analysis

---

## Possible Extensions

* Train a self-supervised model directly on dataset
* Learn fusion instead of concatenation
* Predict traffic metrics directly from embeddings
* Add temporal or geographic features

---

## Author

**Mohammed Ishfaq Mostain**
BSc Computing Science – University of Alberta
GitHub: [https://github.com/Mohammedmostain](https://github.com/Mohammedmostain)

---

## License

This project is provided for educational and research purposes.
