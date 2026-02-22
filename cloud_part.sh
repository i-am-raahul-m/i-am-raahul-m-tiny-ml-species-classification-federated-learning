#!/bin/bash
set -e  # exit immediately on error

echo "=============================================="
echo " FEDERATED EDGE-CLOUD TEST PIPELINE"
echo "=============================================="

# -----------------------------------------------
# GLOBAL PATHS
# -----------------------------------------------
SCRIPT_ROOT="/content/drive/MyDrive/Fog&Edge"
DATASET_ROOT="/content/drive/MyDrive/Fog&Edge/Project"
EDGE_OUTPUTS="$DATASET_ROOT/edge_outputs"
CLOUD_OUTPUTS="$DATASET_ROOT/cloud_outputs"
CLOUD_DATA="$DATASET_ROOT/cloud_data"
INFO_JSON="$EDGE_OUTPUTS/info.json"
TAXON="mammal"

# -----------------------------------------------
# EDGE CONFIG
# -----------------------------------------------
EPOCHS=5
BATCH_SIZE=8
LR=1e-4

# -----------------------------------------------
# CLOUD CONFIG
# -----------------------------------------------
CLOUD_EPOCHS=50
CLOUD_BATCH_SIZE=16
CLOUD_LR=1e-3

# -----------------------------------------------
# STEP 4: CLOUD DATASET CREATION (SAFE)
# -----------------------------------------------
echo ""
echo "[CLOUD | STEP 4] Preparing cloud unlabeled dataset..."

if [ -d "$CLOUD_DATA" ]; then
  echo "✔ cloud_data already exists — skipping creation"
else
  echo "⬇ Creating cloud unlabeled dataset"

  python "$SCRIPT_ROOT/cloud_dataset_maker.py" \
  --val_dirs \
    "$DATASET_ROOT/north_india/train/$TAXON" \
    "$DATASET_ROOT/south_india/train/$TAXON" \
    "$DATASET_ROOT/north_india/val/$TAXON" \
    "$DATASET_ROOT/south_india/val/$TAXON" \
  --cloud_data_dir "$CLOUD_DATA" \
  --max_imgs_per_species 50
fi

# -----------------------------------------------
# STEP 5: CLOUD METADATA PREPROCESSING
# -----------------------------------------------
echo ""
echo "[CLOUD | STEP 5] Aggregating metadata (info.json)..."
python "$SCRIPT_ROOT/cloud_metadata_preprocessor.py" \
  --edge_output_root "$EDGE_OUTPUTS" \
  --output_info_path "$INFO_JSON"

# -----------------------------------------------
# STEP 6: FEDMEAN AGGREGATION
# -----------------------------------------------
echo ""
echo "[CLOUD | STEP 6] Running FedMean aggregation..."
python "$SCRIPT_ROOT/cloud_aggregator_fedmean.py" \
  --edge_dir "$EDGE_OUTPUTS" \
  --save_dir "$CLOUD_OUTPUTS"

# -----------------------------------------------
# STEP 7: LOGIT DISTILLATION
# -----------------------------------------------
echo ""
echo "[CLOUD | STEP 7] Running logit distillation..."
python "$SCRIPT_ROOT/cloud_aggregator_logit_dist.py" \
  --edge_dir "$EDGE_OUTPUTS" \
  --global_backbone "$CLOUD_OUTPUTS/global_shared_backbone.pt" \
  --cloud_data "$CLOUD_DATA" \
  --epochs $CLOUD_EPOCHS \
  --batch_size $CLOUD_BATCH_SIZE \
  --lr $CLOUD_LR \
  --save_dir "$CLOUD_OUTPUTS"

# -----------------------------------------------
# STEP 8: CLOUD VALIDATION
# -----------------------------------------------
echo ""
echo "[CLOUD | STEP 8] Validating global model..."
python "$SCRIPT_ROOT/cloud_aggregator_validation.py" \
  --edge_dir "$EDGE_OUTPUTS" \
  --global_backbone "$CLOUD_OUTPUTS/global_shared_backbone.pt" \
  --cloud_head "$CLOUD_OUTPUTS/cloud_head.pt" \
  --val_dir "$DATASET_ROOT/north_india/val/$TAXON" \
  --batch_size 32

echo ""
echo "=============================================="
echo " PIPELINE COMPLETED SUCCESSFULLY"
echo "=============================================="