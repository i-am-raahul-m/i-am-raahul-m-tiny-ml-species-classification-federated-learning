#!/bin/bash
set -e  # exit immediately on error

echo "=============================================="
echo " FEDERATED EDGE-CLOUD TEST PIPELINE (CLEAN RUN)"
echo "=============================================="

# -----------------------------------------------
# GLOBAL PATHS (EDIT ONCE PER ENVIRONMENT)
# -----------------------------------------------
DATASET_ROOT="/content/drive/MyDrive/Fog&Edge/Project"
EDGE_OUTPUTS="$DATASET_ROOT/edge_outputs"
CLOUD_OUTPUTS="$DATASET_ROOT/cloud_outputs"
CLOUD_DATA="$DATASET_ROOT/cloud_data/"
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
CLOUD_EPOCHS=5
CLOUD_BATCH_SIZE=16
CLOUD_LR=1e-4

# -----------------------------------------------
# STEP 1: REGION-WISE DATASET DOWNLOAD
# -----------------------------------------------
echo ""
echo "[EDGE | STEP 1] Downloading datasets (north & south)..."
python region_wise_dataset_downloader.py \
  --dataset_root "$DATASET_ROOT" \
  --taxon "$TAXON"

# -----------------------------------------------
# STEP 2: TRAIN–VAL SPLIT (CLI VERSION)
# -----------------------------------------------
echo ""
echo "[EDGE | STEP 2] Creating train/val splits..."

python region_dataset_train_val_splitter.py \
  --dataset-root "$DATASET_ROOT" \
  --region all \
  --taxon "$TAXON" \
  --train-ratio 0.8 \
  --seed 42

# -----------------------------------------------
# STEP 3: EDGE TRAINING (SIMULATED CLIENTS)
# -----------------------------------------------
echo ""
echo "[EDGE | STEP 3] Training EDGE model: north_india"
python edge_model.py \
  --dataset_root "$DATASET_ROOT" \
  --edge_output_root "$EDGE_OUTPUTS" \
  --region north_india \
  --taxon "$TAXON" \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR

echo ""
echo "[EDGE | STEP 3] Training EDGE model: south_india"
python edge_model.py \
  --dataset_root "$DATASET_ROOT" \
  --edge_output_root "$EDGE_OUTPUTS" \
  --region south_india \
  --taxon "$TAXON" \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR

# -----------------------------------------------
# STEP 4: CLOUD DATASET CREATION (UNLABELED)
# -----------------------------------------------
echo ""
echo "[CLOUD | STEP 4] Creating cloud unlabeled dataset..."
python cloud_dataset_maker.py \
  --dataset_root "$DATASET_ROOT" \
  --taxon "$TAXON" \
  --output_dir "$CLOUD_DATA" \
  --max_imgs_per_species 5

# -----------------------------------------------
# STEP 5: CLOUD METADATA PREPROCESSING
# -----------------------------------------------
echo ""
echo "[CLOUD | STEP 5] Aggregating metadata (info.json)..."
python cloud_metadata_preprocessor.py \
  --edge_output_root "$EDGE_OUTPUTS" \
  --output_info_path "$INFO_JSON"

# -----------------------------------------------
# STEP 6: FEDMEAN AGGREGATION (GLOBAL BACKBONE)
# -----------------------------------------------
echo ""
echo "[CLOUD | STEP 6] Running FedMean aggregation..."
python cloud_aggregator_fedmean.py \
  --edge_dir "$EDGE_OUTPUTS" \
  --info_json "$INFO_JSON" \
  --save_dir "$CLOUD_OUTPUTS"

# -----------------------------------------------
# STEP 7: LOGIT DISTILLATION (CLOUD HEAD)
# -----------------------------------------------
echo ""
echo "[CLOUD | STEP 7] Running logit distillation..."
python cloud_aggregator_logit_dist.py \
  --edge_dir "$EDGE_OUTPUTS" \
  --info_json "$INFO_JSON" \
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
python cloud_aggregator_validation.py \
  --edge_dir "$EDGE_OUTPUTS" \
  --info_json "$INFO_JSON" \
  --global_backbone "$CLOUD_OUTPUTS/global_shared_backbone.pt" \
  --cloud_head "$CLOUD_OUTPUTS/cloud_head.pt" \
  --val_dir "$DATASET_ROOT/north_india/val/$TAXON" \
  --batch_size 32

echo ""
echo "=============================================="
echo " PIPELINE COMPLETED SUCCESSFULLY"
echo "=============================================="
