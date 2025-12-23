import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import argparse
import os
import json

# =====================================================
# Shared Backbone (MUST MATCH EDGE + AGGREGATOR)
# =====================================================

class SharedBackbone(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        for p in resnet.parameters():
            p.requires_grad = False

        for p in resnet.layer4.parameters():
            p.requires_grad = True

        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1]
        )

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
# Cloud Head (DISTILLED)
# =====================================================

class CloudHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# =====================================================
# Evaluation
# =====================================================

@torch.no_grad()
def evaluate(backbone, head, dataloader, device):
    backbone.eval()
    head.eval()

    correct, total = 0, 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        feats = backbone(images)
        logits = head(feats)

        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


# =====================================================
# Main
# =====================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cloud Validation using FedMean Backbone + Distilled Head"
    )
    parser.add_argument("--edge_dir", type=str, required=True,
                        help="edge_outputs directory (contains info.json)")
    parser.add_argument("--global_backbone", type=str, required=True)
    parser.add_argument("--cloud_head", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------
    # Load metadata (single source of truth)
    # -------------------------------------------------

    info_path = os.path.join(args.edge_dir, "info.json")
    if not os.path.exists(info_path):
        raise RuntimeError("Missing info.json. Run metadata_preprocess.py first.")

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    num_classes = info["global_num_classes"]
    print(f"[INFO] Global classes: {num_classes}")

    # -------------------------------------------------
    # Load models
    # -------------------------------------------------

    backbone = SharedBackbone(embed_dim=256).to(device)
    backbone.load_state_dict(
        torch.load(args.global_backbone, map_location=device),
        strict=True
    )

    head = CloudHead(
        embed_dim=256,
        num_classes=num_classes
    ).to(device)

    head.load_state_dict(
        torch.load(args.cloud_head, map_location=device),
        strict=True
    )

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = ImageFolder(args.val_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    # -------------------------------------------------
    # Evaluate
    # -------------------------------------------------

    acc = evaluate(backbone, head, loader, device)

    print("\nCloud Validation Complete")
    print(f"Validation Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()
