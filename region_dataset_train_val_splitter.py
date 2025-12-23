import argparse
import random
import shutil
from pathlib import Path


def split_species_dir(species_dir, train_dir, val_dir, train_ratio):
    images = list(species_dir.glob("*.jpg"))
    if len(images) < 10:
        return  # skip tiny classes safely

    random.shuffle(images)
    split_idx = int(len(images) * train_ratio)

    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    train_species_dir = train_dir / species_dir.name
    val_species_dir = val_dir / species_dir.name

    train_species_dir.mkdir(parents=True, exist_ok=True)
    val_species_dir.mkdir(parents=True, exist_ok=True)

    for img in train_imgs:
        shutil.move(img, train_species_dir / img.name)
        # shutil.copy(img, train_species_dir / img.name)

    for img in val_imgs:
        shutil.move(img, val_species_dir / img.name)
        # shutil.copy(img, val_species_dir / img.name)


def split_region(dataset_root, region, taxon, train_ratio):
    base_dir = dataset_root / region / taxon
    if not base_dir.exists():
        print(f"[WARN] Missing base dir: {base_dir}")
        return

    train_dir = dataset_root / region / "train" / taxon
    val_dir = dataset_root / region / "val" / taxon

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    species_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    print(f"[{region}] Found {len(species_dirs)} species")

    for species_dir in species_dirs:
        split_species_dir(species_dir, train_dir, val_dir, train_ratio)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Region-wise train/val split for biodiversity datasets"
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root dataset directory (e.g. D:/Fog&Edge)"
    )

    parser.add_argument(
        "--region",
        type=str,
        default="all",
        choices=["north_india", "south_india", "all"],
        help="Region to split"
    )

    parser.add_argument(
        "--taxon",
        type=str,
        required=True,
        help="Taxon name (e.g. mammal)"
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    regions = (
        ["north_india", "south_india"]
        if args.region == "all"
        else [args.region]
    )

    for region in regions:
        print(f"\nSplitting region: {region}")
        split_region(
            dataset_root=args.dataset_root,
            region=region,
            taxon=args.taxon,
            train_ratio=args.train_ratio
        )

    print("\nDataset split completed successfully.")


if __name__ == "__main__":
    main()
