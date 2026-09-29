from pathlib import Path
import json
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
from training.dataset import MEAN, STD


def main():
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "models/resnet18_chess_best.pth"
    if not checkpoint.exists():
        raise FileNotFoundError("Train the model before evaluation.")

    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    classes = data["classes"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(data["model_state_dict"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    ds = datasets.ImageFolder(root / "dataset/val", transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)

    y_true, y_pred = [], []
    with torch.inference_mode():
        for images, labels in loader:
            y_true.extend(labels.tolist())
            y_pred.extend(model(images).argmax(1).tolist())

    output = {
        "classification_report": classification_report(
            y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classes": classes,
    }
    (root / "models/evaluation.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
