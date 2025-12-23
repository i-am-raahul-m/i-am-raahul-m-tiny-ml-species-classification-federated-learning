import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import argparse
import os
import json

# ============================
# Model Definitions
# ============================

class SharedBackbone(nn.Module):
    """
    Federated (shared) part of the model
    ResNet-18 backbone with partial freezing
    """
    def __init__(self, embed_dim=256):
        super().__init__()

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        # Freeze entire backbone
        for p in resnet.parameters():
            p.requires_grad = False

        # Unfreeze last residual block
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


class EdgeClassifierHead(nn.Module):
    def __init__(self, embed_dim=256, num_classes=1):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class EdgeModel(nn.Module):
    def __init__(self, num_classes, embed_dim=256):
        super().__init__()
        self.shared_backbone = SharedBackbone(embed_dim)
        self.head = EdgeClassifierHead(embed_dim, num_classes)

    def forward(self, x):
        return self.head(self.shared_backbone(x))


# ============================
# Training & Evaluation
# ============================

def train_edge(model, dataloader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), 100.0 * correct / total


# ============================
# Metadata Writer
# ============================

def write_metadata(save_dir, region, taxon, dataset: ImageFolder):
    metadata = {
        "region": region,
        "taxon": taxon,
        "num_species": len(dataset.classes),
        "num_samples": len(dataset),
        "species": dataset.classes
    }

    path = os.path.join(save_dir, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Metadata written to {path}")


# ============================
# Main Program (CLI)
# ============================

def main():
    parser = argparse.ArgumentParser(
        description="Edge Training with Federated ResNet-18 Backbone"
    )

    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--edge_output_root", type=str, required=True)
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--taxon", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    # ----------------------------
    # Resolve paths (NO HARDCODE)
    # ----------------------------

    train_dir = os.path.join(
        args.dataset_root, args.region, "train", args.taxon
    )
    val_dir = os.path.join(
        args.dataset_root, args.region, "val", args.taxon
    )
    save_dir = os.path.join(
        args.edge_output_root, args.region
    )

    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============================
    # Transforms
    # ============================

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ============================
    # Datasets & Loaders
    # ============================

    train_ds = ImageFolder(train_dir, transform=transform)
    num_classes = len(train_ds.classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = None
    if os.path.exists(val_dir):
        val_ds = ImageFolder(val_dir, transform=transform)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )

    print(f"[INFO] {args.region}: {num_classes} classes, {len(train_ds)} samples")

    # ============================
    # Model
    # ============================

    model = EdgeModel(
        num_classes=num_classes,
        embed_dim=256
    ).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    # ============================
    # Training Loop
    # ============================

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_edge(
            model, train_loader, optimizer, device
        )

        log = (
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}%"
        )

        if val_loader:
            val_loss, val_acc = evaluate(
                model, val_loader, device
            )
            log += (
                f" || Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )

        print(log)

    # ============================
    # Save Outputs (CLOUD EXPECTS THESE)
    # ============================

    torch.save(
        model.shared_backbone.state_dict(),
        os.path.join(save_dir, "shared_backbone.pt")
    )
    torch.save(
        model.head.state_dict(),
        os.path.join(save_dir, "edge_head.pt")
    )

    write_metadata(
        save_dir=save_dir,
        region=args.region,
        taxon=args.taxon,
        dataset=train_ds
    )

    print("\nTraining complete.")
    print(f"- saved to {save_dir}")


if __name__ == "__main__":
    main()
