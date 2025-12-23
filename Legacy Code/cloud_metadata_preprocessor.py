import os
import json

# ============================
# HARD-CODED PIPELINE PATHS
# ============================

EDGE_OUTPUT_ROOT = "D:/Fog&Edge/edge_outputs"
OUTPUT_INFO_PATH = os.path.join(EDGE_OUTPUT_ROOT, "info.json")


def main():
    global_species_set = set()
    total_samples = 0
    taxon = None

    clients_info = {}

    # ----------------------------
    # Iterate over edge clients
    # ----------------------------

    for client in os.listdir(EDGE_OUTPUT_ROOT):
        client_dir = os.path.join(EDGE_OUTPUT_ROOT, client)

        if not os.path.isdir(client_dir):
            continue

        metadata_path = os.path.join(client_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            print(f"[WARN] Missing metadata.json for {client}")
            continue

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Sanity: taxon must match across all clients
        if taxon is None:
            taxon = metadata["taxon"]
        elif taxon != metadata["taxon"]:
            raise RuntimeError(
                f"Taxon mismatch: expected {taxon}, found {metadata['taxon']}"
            )

        species = metadata["species"]
        num_samples = metadata["num_samples"]

        global_species_set.update(species)
        total_samples += num_samples

        clients_info[client] = {
            "num_samples": num_samples,
            "num_species": metadata["num_species"]
        }

        print(
            f"[INFO] Loaded {client}: "
            f"{metadata['num_species']} species, "
            f"{num_samples} samples"
        )

    # ----------------------------
    # Build final info.json
    # ----------------------------

    info = {
        "taxon": taxon,
        "global_num_classes": len(global_species_set),
        "global_species": sorted(global_species_set),
        "total_samples": total_samples,
        "clients": clients_info
    }

    with open(OUTPUT_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("\n[OK] Cloud metadata prepared")
    print(f"- Global classes: {info['global_num_classes']}")
    print(f"- Total samples: {info['total_samples']}")
    print(f"- Written to: {OUTPUT_INFO_PATH}")


if __name__ == "__main__":
    main()
