import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse
import os
import json
from glob import glob

# =====================================================
# Shared Backbone (MUST MATCH EDGE + FEDMEAN)
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
# Heads
# =====================================================

class CloudHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# =====================================================
# Unlabeled Dataset
# =====================================================

class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.paths = sorted(glob(os.path.join(root_dir, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# =====================================================
# Logit Distillation
# =====================================================

def distillation_loss(student_logits, teacher_logits, T=2.0):
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean"
    ) * (T * T)


def train_distillation(
    backbone,
    cloud_head,
    edge_heads,                 # dict: client -> head
    client_meta,                # dict from info.json
    global_class_to_index,      # dict
    global_num_classes,         # int
    dataloader,
    optimizer,
    device,
    temperature=2.0
):
    backbone.eval()
    cloud_head.train()
    for h in edge_heads.values():
        h.eval()

    total_loss = 0.0

    for images in dataloader:
        images = images.to(device)

        with torch.no_grad():
            features = backbone(images)
            all_teacher_logits = []

            for client, meta in client_meta.items():
                head = edge_heads[client]
                local_logits = head(features)  # [B, local_classes]

                expanded = torch.zeros(
                    features.size(0),
                    global_num_classes,
                    device=device
                )

                for species, local_idx in meta["class_to_index"].items():
                    global_idx = global_class_to_index[species]
                    expanded[:, global_idx] = local_logits[:, local_idx]

                all_teacher_logits.append(expanded)

            teacher_logits = torch.stack(
                all_teacher_logits, dim=0
            ).mean(dim=0)

        student_logits = cloud_head(features)

        loss = distillation_loss(
            student_logits,
            teacher_logits,
            T=temperature
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# =====================================================
# Main
# =====================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cloud Logit Distillation (metadata-driven)"
    )
    parser.add_argument("--edge_dir", type=str, required=True,
                        help="edge_outputs directory")
    parser.add_argument("--global_backbone", type=str, required=True)
    parser.add_argument("--cloud_data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="./cloud_outputs")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =================================================
    # Load metadata (SINGLE SOURCE OF TRUTH)
    # =================================================

    info_path = os.path.join(args.edge_dir, "info.json")
    if not os.path.exists(info_path):
        raise RuntimeError("Missing info.json. Run metadata_preprocess.py first.")

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    global_num_classes = info["global_num_classes"]
    global_class_to_index = info["global_class_to_index"]
    client_meta = info["clients"]

    print(f"[INFO] Global classes: {global_num_classes}")

    # =================================================
    # Load Backbone
    # =================================================

    backbone = SharedBackbone(embed_dim=256).to(device)
    backbone.load_state_dict(
        torch.load(args.global_backbone, map_location=device),
        strict=True
    )

    # =================================================
    # Load Edge Heads (teachers)
    # =================================================

    edge_heads = {}

    for client, meta in client_meta.items():
        head_path = os.path.join(
            args.edge_dir, client, "edge_head.pt"
        )

        if not os.path.exists(head_path):
            raise RuntimeError(f"Missing edge_head.pt for {client}")

        head = CloudHead(
            embed_dim=256,
            num_classes=meta["num_classes"]
        ).to(device)

        head.load_state_dict(
            torch.load(head_path, map_location=device),
            strict=True
        )

        edge_heads[client] = head
        print(f"Loaded {client} head ({meta['num_classes']} classes)")

    # =================================================
    # Dataset
    # =================================================

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = UnlabeledImageDataset(args.cloud_data, transform)

    if len(dataset) == 0:
        raise RuntimeError("Cloud distillation dataset is empty.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )

    # =================================================
    # Student Head (GLOBAL)
    # =================================================

    cloud_head = CloudHead(
        embed_dim=256,
        num_classes=global_num_classes
    ).to(device)

    optimizer = torch.optim.Adam(
        cloud_head.parameters(), lr=args.lr
    )

    # =================================================
    # Distillation Loop
    # =================================================

    for epoch in range(args.epochs):
        loss = train_distillation(
            backbone,
            cloud_head,
            edge_heads,
            client_meta,
            global_class_to_index,
            global_num_classes,
            loader,
            optimizer,
            device
        )

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Distillation Loss: {loss:.4f}"
        )

    # =================================================
    # Save
    # =================================================

    torch.save(
        cloud_head.state_dict(),
        os.path.join(args.save_dir, "cloud_head.pt")
    )

    print("\nLogit distillation complete.")
    print("- cloud_head.pt (global student)")


if __name__ == "__main__":
    main()
