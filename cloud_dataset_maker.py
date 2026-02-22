import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare flat unlabeled cloud dataset from datasets"
    )
    parser.add_argument(
        "--val_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Directories that directly contain species folders"
    )
    parser.add_argument(
        "--cloud_data_dir",
        type=str,
        required=True,
        help="Output directory for flat unlabeled cloud dataset"
    )
    parser.add_argument(
        "--max_imgs_per_species",
        type=int,
        default=20,
        help="Max images per species to copy"
    )

    args = parser.parse_args()

    val_dirs = [Path(p) for p in args.val_dirs]
    cloud_data_dir = Path(args.cloud_data_dir)
    max_imgs = args.max_imgs_per_species

    cloud_data_dir.mkdir(parents=True, exist_ok=True)

    for val_root in val_dirs:
        if not val_root.exists():
            print(f"[WARN] Missing: {val_root}")
            continue

        region_name = val_root.parents[2].name  # north_india / south_india
        split_name = val_root.parents[1].name   # train / val

        print(f"Processing {val_root}")

        for species_dir in sorted(
            d for d in val_root.iterdir() if d.is_dir()
        ):
            images = sorted(species_dir.glob("*.jpg"))[:max_imgs]

            for img in images:
                dst_name = (
                    f"{region_name}_{split_name}_"
                    f"{species_dir.name}_{img.name}"
                )
                dst_path = cloud_data_dir / dst_name

                if not dst_path.exists():
                    shutil.copy2(img, dst_path)

    print("\nFlat unlabeled cloud dataset prepared successfully.")
    print(f"Images stored in: {cloud_data_dir}")


if __name__ == "__main__":
    main()
