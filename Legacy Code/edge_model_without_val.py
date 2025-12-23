import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import argparse
import os

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

        # -------- Freeze entire backbone --------
        for p in resnet.parameters():
            p.requires_grad = False

        # -------- Unfreeze last residual block (layer4) --------
        for p in resnet.layer4.parameters():
            p.requires_grad = True

        # Remove classifier
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1]
        )  # [B, 512, 1, 1]

        # Projection head (shared)
        self.projection = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.projection(x)
        return x


class EdgeClassifierHead(nn.Module):
    """
    Edge-specific classifier head (NOT federated)
    """
    def __init__(self, embed_dim=256, num_classes=20):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class EdgeModel(nn.Module):
    """
    Complete edge model
    """
    def __init__(self, num_classes=20, embed_dim=256):
        super().__init__()
        self.shared_backbone = SharedBackbone(embed_dim)
        self.head = EdgeClassifierHead(embed_dim, num_classes)

    def forward(self, x):
        features = self.shared_backbone(x)
        logits = self.head(features)
        return logits


# ============================
# Training Function
# ============================

def train_edge(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100.0 * correct / total
    avg_loss = total_loss / len(dataloader)

    return avg_loss, acc


# ============================
# Main Program
# ============================

def main():
    parser = argparse.ArgumentParser(
        description="Edge Training with Federated ResNet-18 Backbone"
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="./edge_outputs")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============================
    # Dataset & Dataloader
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

    dataset = ImageFolder(args.data_dir, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # edge safety
        pin_memory=False
    )

    # ============================
    # Model Initialization
    # ============================

    model = EdgeModel(
        num_classes=args.num_classes,
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
        loss, acc = train_edge(
            model, dataloader, optimizer, device
        )
        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Loss: {loss:.4f} | Accuracy: {acc:.2f}%"
        )

    # ============================
    # Save Federated & Local Parts
    # ============================

    torch.save(
        model.shared_backbone.state_dict(),
        os.path.join(args.save_dir, "shared_backbone.pt")
    )

    torch.save(
        model.head.state_dict(),
        os.path.join(args.save_dir, "edge_head.pt")
    )

    print("\nTraining complete.")
    print("Saved:")
    print("- shared_backbone.pt (to be federated)")
    print("- edge_head.pt (local only)")


if __name__ == "__main__":
    main()
