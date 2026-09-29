import os
import torch
from torchvision import datasets, transforms

def get_data_loaders(data_dir="dataset", batch_size=32):
    """
    Creates PyTorch DataLoaders for training our ResNet-18 piece classifier.
    Includes data augmentation to handle class imbalance.
    """
    
    # 1. Define transformations with augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),      # Random flip
        transforms.RandomRotation(10),                # Random rotation +/- 10 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Random color variations
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 2. Validation/test transforms (no augmentation, just resize/normalize)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    # 3. Load datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    
    # 4. Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )

    print(f"[*] Dataset loaded!")
    print(f"    Train: {len(train_dataset)} images")
    print(f"    Val: {len(val_dataset)} images")
    print(f"    Classes: {train_dataset.classes}\n")
    
    return train_loader, val_loader

if __name__ == "__main__":
    print("Testing data loader setup...")
    train_loader, val_loader = get_data_loaders()
