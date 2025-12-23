import torch
import torch.nn as nn
from torchvision import models
import argparse
import os
import json

# =====================================================
# Shared Backbone (MUST MATCH EDGE SCRIPT EXACTLY)
# =====================================================

class SharedBackbone(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        # Freeze everything
        for p in resnet.parameters():
            p.requires_grad = False

        # Unfreeze only layer4
        for p in resnet.layer4.parameters():
            p.requires_grad = True

        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1]
        )  # [B, 512, 1, 1]

        self.projection = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.projection(x)


# =====================================================
# Weighted FedMean
# =====================================================

def fedavg_weighted(client_states, client_sizes):
    total_samples = sum(client_sizes)
    avg_state = {}

    for key in client_states[0]:
        avg_state[key] = sum(
            client_states[i][key] * client_sizes[i]
            for i in range(len(client_states))
        ) / total_samples

    return avg_state


# =====================================================
# Main Aggregator
# =====================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cloud FedMean Aggregator (metadata-driven)"
    )
    parser.add_argument("--edge_dir", type=str, required=True,
                        help="Directory with edge client folders")
    parser.add_argument("--save_dir", type=str, default="./cloud_outputs")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =================================================
    # Load aggregated metadata (SINGLE SOURCE OF TRUTH)
    # =================================================

    info_path = os.path.join(args.edge_dir, "info.json")
    if not os.path.exists(info_path):
        raise RuntimeError("Missing info.json. Run metadata_preprocess.py first.")

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    client_metadata = info["clients"]

    # =================================================
    # Load Edge Client Backbones
    # =================================================

    client_states = []
    client_sizes = []

    print(f"Found {len(client_metadata)} edge clients")

    for client, meta in client_metadata.items():
        backbone_path = os.path.join(
            args.edge_dir, client, "shared_backbone.pt"
        )

        if not os.path.exists(backbone_path):
            raise RuntimeError(f"Missing backbone for {client}")

        state = torch.load(backbone_path, map_location="cpu")
        client_states.append(state)

        n = meta["num_samples"]
        client_sizes.append(n)

        print(f"Client {client}: {n} samples")

    # =================================================
    # FedMean Aggregation
    # =================================================

    global_state = fedavg_weighted(client_states, client_sizes)

    global_backbone = SharedBackbone(embed_dim=256).to(device)
    global_backbone.load_state_dict(global_state, strict=True)

    # =================================================
    # Save Global Backbone
    # =================================================

    torch.save(
        global_backbone.state_dict(),
        os.path.join(args.save_dir, "global_shared_backbone.pt")
    )

    print("\nFederated aggregation complete.")
    print("Saved:")
    print("- global_shared_backbone.pt (send to edges)")


if __name__ == "__main__":
    main()
