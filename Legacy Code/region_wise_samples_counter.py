import os
import csv

taxon = "mammal"
DATASET_ROOT = "D:/Fog&Edge/"
EDGE_OUTPUT_ROOT = "D:/Fog&Edge/edge_outputs"
OUTPUT_CSV = "region_wise_classes.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

results = []

for region in ("north_india", "south_india"):
    base_path = os.path.join(DATASET_ROOT, region, "train", taxon)

    species_set = set()
    num_samples = 0

    if not os.path.exists(base_path):
        print(f"[WARN] Path does not exist: {base_path}")
        results.append({
            "region": region,
            "num_species": 0,
            "num_samples": 0
        })
        continue

    for root, dirs, files in os.walk(base_path):
        # First-level directories under taxon are species
        if root == base_path:
            species_set.update(dirs)

        # Count image files across all species folders
        for file in files:
            if os.path.splitext(file.lower())[1] in IMAGE_EXTENSIONS:
                num_samples += 1

    # -------------------------------------------------
    # WRITE num_samples.txt WHERE edge_model.py EXPECTS
    # -------------------------------------------------

    client_dir = os.path.join(EDGE_OUTPUT_ROOT, region)
    os.makedirs(client_dir, exist_ok=True)

    samples_txt_path = os.path.join(client_dir, "num_samples.txt")
    with open(samples_txt_path, "w", encoding="utf-8") as f:
        f.write(str(num_samples))

    results.append({
        "region": region,
        "num_species": len(species_set),
        "num_samples": num_samples
    })

# Write CSV
with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["region", "num_species", "num_samples"]
    )
    writer.writeheader()
    writer.writerows(results)

print("Region-wise species and sample count written to:", OUTPUT_CSV)
for r in results:
    print(
        f"{r['region']}: "
        f"{r['num_species']} species, "
        f"{r['num_samples']} samples"
    )
