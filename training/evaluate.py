import torch
import torch.nn as nn
from torchvision import models
from dataset import get_data_loaders
import os

def evaluate_model():
    print("🔍 Initializing Chess Vision Model Evaluation...")

    # 1. Get our validation data loader from dataset.py
    _, val_loader = get_data_loaders(data_dir="dataset", batch_size=32)

    # 2. Recreate the ResNet-18 architecture (13 classes)
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 13)

    # 3. Load your trained model weights
    model_path = "models/resnet18_chess.pth"
    if not os.path.exists(model_path):
        print(f"❌ Error: Model weights not found at {model_path}. Run training/train.py first!")
        return

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    
    # 4. Set device and switch model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # 5. Evaluation Loop (no gradient calculations needed)
    print("Evaluating model performance on validation set...")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            # Get predicted class with highest probability
            _, preds = torch.max(outputs, 1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

    # 6. Calculate final metrics
    avg_loss = total_loss / total_samples
    accuracy = (correct_predictions / total_samples) * 100

    print("\n📊 --- Evaluation Results ---")
    print(f"Total Validation Images: {total_samples}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.2f}%")
    print("----------------------------\n")

if __name__ == "__main__":
    evaluate_model()