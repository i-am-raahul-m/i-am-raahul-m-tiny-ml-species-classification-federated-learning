import os
import json
import argparse

# ============================
# Main
# ============================

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate edge metadata into global cloud info.json"
    )
    parser.add_argument(
        "--edge_output_root",
        type=str,
        required=True,
        help="Path to edge_outputs directory"
    )
    parser.add_argument(
        "--output_info_path",
        type=str,
        required=True,
        help="Path to write combined info.json"
    )

    args = parser.parse_args()

    edge_output_root = args.edge_output_root
    output_info_path = args.output_info_path

    global_species_set = set()
    total_samples = 0
    taxon = None
    clients_info = {}

    # ----------------------------
    # Iterate over edge clients
    # ----------------------------

    for client in os.listdir(edge_output_root):
        client_dir = os.path.join(edge_output_root, client)

        if not os.path.isdir(client_dir):
            continue

        metadata_path = os.path.join(client_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            print(f"[WARN] Missing metadata.json for {client}")
            continue

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Taxon consistency
        if taxon is None:
            taxon = metadata["taxon"]
        elif taxon != metadata["taxon"]:
            raise RuntimeError(
                f"Taxon mismatch: expected {taxon}, found {metadata['taxon']}"
            )

        species = metadata["species"]
        num_samples = metadata["num_samples"]
        num_classes = metadata["num_species"]
        class_to_index = metadata["class_to_index"]  # 🔥 REQUIRED

        global_species_set.update(species)
        total_samples += num_samples

        # 🔥 FINAL CONTRACT (NO MISSING KEYS)
        clients_info[client] = {
            "num_samples": num_samples,
            "num_classes": num_classes,
            "class_to_index": class_to_index
        }

        print(
            f"[INFO] Loaded {client}: "
            f"{num_classes} classes, "
            f"{num_samples} samples"
        )

    # ----------------------------
    # Global class index
    # ----------------------------

    global_species = sorted(global_species_set)
    global_class_to_index = {
        name: idx for idx, name in enumerate(global_species)
    }

    # ----------------------------
    # Build final info.json
    # ----------------------------

    info = {
        "taxon": taxon,
        "global_num_classes": len(global_species),
        "global_species": global_species,
        "global_class_to_index": global_class_to_index,
        "total_samples": total_samples,
        "clients": clients_info
    }

    os.makedirs(os.path.dirname(output_info_path), exist_ok=True)

    with open(output_info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("\n[OK] Cloud metadata prepared")
    print(f"- Global classes: {info['global_num_classes']}")
    print(f"- Total samples: {info['total_samples']}")
    print(f"- Written to: {output_info_path}")


if __name__ == "__main__":
    main()
