from __future__ import annotations

from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(image_size=224):
    train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    evaluation = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train, evaluation


def get_data_loaders(data_dir, batch_size=32, num_workers=0):
    data_dir = Path(data_dir)
    train_tf, val_tf = get_transforms()
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=val_tf)

    if train_ds.classes != val_ds.classes:
        raise ValueError("Train and validation class sets differ.")

    counts = torch.bincount(torch.tensor(train_ds.targets), minlength=len(train_ds.classes))
    if (counts == 0).any():
        missing = [train_ds.classes[i] for i, n in enumerate(counts) if n == 0]
        raise ValueError(f"Missing training examples: {missing}")

    class_weights = 1.0 / counts.float()
    sample_weights = torch.tensor([class_weights[t] for t in train_ds.targets], dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    kwargs = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    return (
        DataLoader(train_ds, batch_size=batch_size, sampler=sampler, **kwargs),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kwargs),
        train_ds.classes,
    )
