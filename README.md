# Federated Edge–Cloud Species Classification Pipeline

This repository implements a **Federated Learning (FL)** system designed for species classification. It enables edge devices to train on regional data while keeping labels and classifiers private, sharing only a common backbone with a central cloud aggregator.



---

## Overview

The pipeline utilizes a split-model architecture:
* **Edge Devices:** Train a local classifier on regional data, fine-tuning only the shared backbone (ResNet-18).
* **Cloud Aggregator:** Performs **FedAvg** (Federated Averaging) on the received backbones to produce a global feature extractor.

### Logical Stages
1. **Dataset Creation:** (Completed)
2. **Edge Training:** `edge_model.py`
3. **Cloud Aggregation:** `cloud_aggregator.py`

---

## 1. Directory Structure

The dataset must be organized by region and taxon to facilitate local training:

```text
DATASET_ROOT/
├── north_india/
│   └── mammal/
│       ├── SpeciesA-LatinA/
│       ├── SpeciesB-LatinB/
│       └── ...
└── south_india/
    └── mammal/
        ├── SpeciesX-LatinX/
        └── ...
```

---

## 2. Edge Training (`edge_model.py`)

**Purpose:** Trains an edge-specific classifier and saves the shared backbone for federation.

### Sample Command
Run this on each edge device:

```bash
python edge_model.py \
  --data_dir "D:/Fog&Edge/south_india/mammal" \
  --epochs 5 \
  --batch_size 8 \
  --lr 1e-4 \
  --num_classes 20 \
  --save_dir "./edge_outputs/client_south"
```

### Output Files
Located in `./edge_outputs/client_south/`:

| File | Purpose | Sent to Cloud? |
| :--- | :--- | :--- |
| `shared_backbone.pt` | Federated backbone (Weights) | ✅ YES |
| `edge_head.pt` | Local classifier (Private) | ❌ NO |
| `num_samples.txt` | Total training samples | ✅ YES |

> **Note:** `num_samples.txt` should contain a single integer (e.g., `1247`) representing the local sample count for weighted averaging.

---

## 3. Preparing Cloud Uploads

The cloud server expects all client contributions to be organized into a single directory:

```text
EDGE_UPLOADS/
├── client_north/
│   ├── shared_backbone.pt
│   └── num_samples.txt
├── client_south/
│   ├── shared_backbone.pt
│   └── num_samples.txt
└── client_X/
    ├── shared_backbone.pt
    └── num_samples.txt
```

---

## 4. Cloud Aggregation (`cloud_aggregator.py`)

**Purpose:** Performs sample-weighted FedAvg to create a global backbone.

### Sample Command
```bash
python cloud_aggregator.py \
  --edge_dir "./EDGE_UPLOADS" \
  --save_dir "./cloud_outputs"
```

### Output
The aggregator produces:
* `./cloud_outputs/global_shared_backbone.pt`: **Redistribute this to all edge devices.**

---

## 5. Updating Edge Models

Upon receiving the global backbone, edge devices must update their local models:

```python
# Loading the global backbone into the edge model
model.shared_backbone.load_state_dict(
    torch.load("global_shared_backbone.pt")
)
```

---

## 6. Flow of One Complete Federated Round

1. **Edge:** Trains locally -> Saves `shared_backbone.pt` and `num_samples.txt`.
2. **Cloud:** Collects all edge folders -> Runs `cloud_aggregator.py`.
3. **Cloud:** Produces and sends `global_shared_backbone.pt` back to all edges.
4. **Edge:** Loads global backbone -> Resumes training or inference.