import shutil
from pathlib import Path

# ============================
# CONFIG
# ============================

VAL_DIRS = [
    Path("D:/Fog&Edge/north_india"),
    Path("D:/Fog&Edge/south_india")
]

CLOUD_DATA_DIR = Path("D:/Fog&Edge/cloud_dataset")
TAXON = "mammal"
MAX_IMGS_PER_SPECIES = 5

# ============================
# SCRIPT
# ============================

CLOUD_DATA_DIR.mkdir(parents=True, exist_ok=True)

for val_root in VAL_DIRS:
    taxon_dir = val_root / TAXON
    region_name = val_root.parent.name

    if not taxon_dir.exists():
        print(f"[WARN] Missing: {taxon_dir}")
        continue

    print(f"Processing {taxon_dir}")

    for species_dir in sorted(d for d in taxon_dir.iterdir() if d.is_dir()):
        images = sorted(species_dir.glob("*.jpg"))[:MAX_IMGS_PER_SPECIES]

        for img in images:
            dst_name = f"{region_name}_{species_dir.name}_{img.name}"
            dst_path = CLOUD_DATA_DIR / dst_name

            if not dst_path.exists():
                shutil.copy2(img, dst_path)

print("\nFlat unlabeled cloud dataset prepared successfully.")
print(f"Images stored in: {CLOUD_DATA_DIR}")
