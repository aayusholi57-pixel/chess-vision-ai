import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

def predict_square(image_path):
    # 1. Define the exact same preprocessing transforms used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 2. Load our 13 class names in the exact alphabetical order PyTorch assigns them
    # (ImageFolder sorts folders alphabetically)
    classes = [
        "black_bishop", "black_king", "black_knight", "black_pawn", 
        "black_queen", "black_rook", "empty", 
        "white_bishop", "white_king", "white_knight", "white_pawn", 
        "white_queen", "white_rook"
    ]

    # 3. Load the model architecture and attach our 13-class final layer
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 13)

    # 4. Load the trained weights we saved in our models/ folder
    model_path = "models/resnet18_chess.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}. Run train.py first!")
        return

    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode (turns off training behaviors like dropout)

    # 5. Open and transform the target square image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add a batch dimension [1, 3, 224, 224]

    # 6. Run Inference!
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get the highest scoring class index
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    predicted_class = classes[predicted_idx.item()]
    print(f"🔍 Prediction for {image_path}:")
    print(f"   Chess Piece: **{predicted_class}**")
    print(f"   Confidence:  {confidence.item() * 100:.2f}%\n")

if __name__ == "__main__":
    # Test with one of your cropped squares from experiments/squares/
    # (Make sure to point to an actual image you sliced earlier!)
    sample_image = "E:\\AI-ML\\chess-vision-ai\\experiments\\experiments\\squares\\square_r0_c0.jpg"
    if os.path.exists(sample_image):
        predict_square(sample_image)
    else:
        print(f"Please place a square image at {sample_image} to test inference.")