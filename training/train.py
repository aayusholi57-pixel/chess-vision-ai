import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights
from dataset import get_data_loaders
import os

def train_model(epochs=10):
    print("[*] Initializing ResNet-18 Chess Piece Classifier...")

    # Change to parent directory to find dataset
   # 1. Dynamically find the project root and dataset folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "dataset")

    # 2. Get our Train DataLoaders using the absolute path
    train_loader, val_loader = get_data_loaders(data_dir=data_path, batch_size=32)

    # 2. Load ResNet-18 with pre-trained ImageNet weights
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # 3. Freeze early layers (transfer learning)
    for param in model.parameters():
        param.requires_grad = False

    # 4. Replace the final classification layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 13)

    # 5. Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 6. Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[*] Using device: {device}")

    # 7. Complete Training Loop
    print(f"[*] Starting training for {epochs} epochs...\n")

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Zero gradients, forward pass, calculate loss, backward pass, update
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("models", exist_ok=True)
            best_model_path = "models/resnet18_chess_best.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"    [+] Best model saved!")

    print("\n[*] Training complete!")

    # 8. Save the final trained model weights
    os.makedirs("models", exist_ok=True)
    model_path = "models/resnet18_chess.pth"
    torch.save(model.state_dict(), model_path)
    print(f"[+] Final model saved to {model_path}")

if __name__ == "__main__":
    train_model(epochs=10)

def train_model(epochs=10):
    print("[*] Initializing ResNet-18 Chess Piece Classifier...")

    # Get the directory where train.py is located (.../training)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the project root (.../chess-vision-ai)
    project_root = os.path.dirname(current_dir)
    
    # Define the exact path to the dataset folder
    data_path = os.path.join(project_root, "dataset")

    # 1. Get our Train DataLoaders using the absolute path
    train_loader, val_loader = get_data_loaders(data_dir=data_path, batch_size=32)

    # 2. Load ResNet-18 with pre-trained ImageNet weights
    # ... (rest of your code remains exactly the same)