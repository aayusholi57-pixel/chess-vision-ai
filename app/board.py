import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)

CLASSES = [
    "black_bishop", "black_king", "black_knight", "black_pawn", 
    "black_queen", "black_rook", "empty", 
    "white_bishop", "white_king", "white_knight", "white_pawn", 
    "white_queen", "white_rook"
]

_model_cache = None
_model_trained = False

def load_chess_model():
    """Loads our trained ResNet-18 model for inference with caching."""
    global _model_cache, _model_trained
    
    if _model_cache is not None:
        return _model_cache, CLASSES, _model_trained
    
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    
    model_path = "models/resnet18_chess.pth"
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            logger.info(f"Loaded trained model from {model_path}")
            _model_trained = True
        except Exception as e:
            logger.warning(f"Failed to load model weights: {e}. Using untrained model.")
            _model_trained = False
    else:
        logger.warning(f"Model file not found at {model_path}. Using untrained model. "
                      f"Please train and save your model to {model_path}")
        _model_trained = False
    
    model.eval()
    _model_cache = model
    return model, CLASSES, _model_trained

def reconstruct_board_from_squares(squares_dir="experiments/squares"):
    """
    Loops through all 64 square images, predicts the piece on each, 
    and builds an 8x8 board matrix.
    """
    model, classes, is_trained = load_chess_model()
    
    if not is_trained:
        logger.warning("Model is not trained. Predictions will be random/unreliable.")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    board_matrix = [["" for _ in range(8)] for _ in range(8)]

    for row in range(8):
        for col in range(8):
            square_path = os.path.join(squares_dir, f"square_r{row}_c{col}.jpg")
            if not os.path.exists(square_path):
                board_matrix[row][col] = "empty"
                continue
            
            try:
                image = Image.open(square_path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted_idx = torch.max(outputs[0], 0)
                    
                board_matrix[row][col] = classes[predicted_idx.item()]
            except Exception as e:
                logger.error(f"Error processing square {row},{col}: {e}")
                board_matrix[row][col] = "empty"

    return board_matrix

if __name__ == "__main__":
    print("Reconstructing board matrix from 64 squares...")
    matrix = reconstruct_board_from_squares()
    for row in matrix:
        print(row)
