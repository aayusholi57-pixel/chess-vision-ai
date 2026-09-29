from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from training.dataset import get_data_loaders


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(num_classes):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total


def train(args):
    seed_everything(args.seed)
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True)

    train_loader, val_loader, classes = get_data_loaders(
        root / "dataset", args.batch_size, args.workers
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Stage 1: train the classification head while preserving ImageNet features.
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=args.head_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.warmup_epochs, 1)
    )

    best_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        if epoch == args.warmup_epochs + 1:
            # Stage 2: fine-tune the final residual block and classifier at lower LR.
            for parameter in model.layer4.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.layer4.parameters(), "lr": args.finetune_lr},
                    {"params": model.fc.parameters(), "lr": args.head_lr * 0.1},
                ],
                weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(args.epochs - args.warmup_epochs, 1)
            )

        model.train()
        loss_sum = correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        record = {
            "epoch": epoch,
            "stage": "head" if epoch <= args.warmup_epochs else "fine_tune_layer4",
            "train_loss": loss_sum / total,
            "train_accuracy": correct / total,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
        }
        history.append(record)
        print(json.dumps(record))

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "architecture": "resnet18",
                    "image_size": 224,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                model_dir / "resnet18_chess_best.pth",
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "architecture": "resnet18",
            "image_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        model_dir / "resnet18_chess.pth",
    )
    (model_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"Training complete on {device}. Best validation loss: {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())
