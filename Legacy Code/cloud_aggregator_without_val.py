import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import argparse
import os
from glob import glob

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
# Weighted FedAvg (TRUE FedAvg)
# =====================================================

def fedavg_weighted(client_states, client_sizes):
    """
    client_states: list of state_dicts
    client_sizes: list of sample counts per client
    """
    total_samples = sum(client_sizes)
    avg_state = {}

    for key in client_states[0]:
        avg_state[key] = sum(
            (client_states[i][key] * client_sizes[i])
            for i in range(len(client_states))
        ) / total_samples

    return avg_state


# =====================================================
# Main Aggregator
# =====================================================

def main():
    parser = argparse.ArgumentParser(
        description="Correct Cloud FedAvg Aggregator (Edge-Compatible)"
    )
    parser.add_argument("--edge_dir", type=str, required=True,
                        help="Directory with edge client folders")
    parser.add_argument("--save_dir", type=str, default="./cloud_outputs")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # =================================================
    # Load Edge Client Models
    # =================================================

    client_dirs = [
        d for d in os.listdir(args.edge_dir)
        if os.path.isdir(os.path.join(args.edge_dir, d))
    ]

    if not client_dirs:
        raise RuntimeError("No edge client directories found.")

    client_states = []
    client_sizes = []

    print(f"Found {len(client_dirs)} edge clients")

    for client in client_dirs:
        backbone_path = os.path.join(
            args.edge_dir, client, "shared_backbone.pt"
        )
        size_path = os.path.join(
            args.edge_dir, client, "num_samples.txt"
        )

        if not os.path.exists(backbone_path):
            raise RuntimeError(f"Missing backbone for {client}")

        state = torch.load(backbone_path, map_location="cpu")
        client_states.append(state)

        # ---- REQUIRED FOR TRUE FedAvg ----
        if os.path.exists(size_path):
            with open(size_path) as f:
                n = int(f.read().strip())
        else:
            n = 1  # fallback (still valid)
        client_sizes.append(n)

        print(f"Client {client}: {n} samples")

    # =================================================
    # Aggregate
    # =================================================

    global_state = fedavg_weighted(
        client_states,
        client_sizes
    )

    # =================================================
    # Load into Global Backbone (STRICT)
    # =================================================

    global_backbone = SharedBackbone(embed_dim=256)
    global_backbone.load_state_dict(global_state, strict=True)

    # =================================================
    # Save Global Model
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
