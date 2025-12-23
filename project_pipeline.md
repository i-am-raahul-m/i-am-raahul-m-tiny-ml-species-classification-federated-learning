# EDGE

## RUN_ONLY_ONCE_IF_PERSISTENT_STORAGE {
## 1. region_wise_dataset_downloader.py
INPUTS: No inputs

OUTPUTS: dataset stored "D:/Fog&Edge/north_india/mammals", "D:/Fog&Edge/south_india/mammals"

## 2. region_dataset_train_val_splitter.py
INPUTS: dataset stored

OUTPUTS: train-val split without affecting existing dataset.

## 3. edge_model.py
INPUTS: --region, --epochs, --batch_size, --lr, --num_classes

OUTPUTS: Entire "D:/Fog&Edge/edge_outputs"

# CLOUD

## RUN_ONLY_ONCE_IF_PERSISTENT_STORAGE {
## 1. cloud_dataset_maker.py
INPUTS: dataset stored

OUTPUTS: cloud logit distillation dataset
}

## 2. cloud_metadata_preprocessor:
INPUTS: various regions metadata.json

OUTPUTS: info.json

## 3. cloud_aggregator_fedmean.py
INPUTS: info.json, --edge_dir ("D:/Fog&Edge/edge_outputs"), --save_dir

OUTPUTS: save_dir / "global_shared_backbone.pt"

## 4. cloud_aggregator_logit_dist.py
INPUTS: info.json, --edge_dir,  --global_backbone, --cloud_data, --epochs, --batch_size, --lr,--save_dir

OUTPUTS: save_dir / "cloud_head.pt"

## 5. cloud_aggregator_validation.py
INPUTS: info.json, --edge_dir, --global_backbone, --cloud_head, --val_dir, --batch_size

OUTPUTS: validation results (in terminal)