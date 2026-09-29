from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

from app.config import MIN_CONFIDENCE, MODEL_PATH

logger = logging.getLogger(__name__)

CLASSES = [
    "black_bishop", "black_king", "black_knight", "black_pawn",
    "black_queen", "black_rook", "empty",
    "white_bishop", "white_king", "white_knight", "white_pawn",
    "white_queen", "white_rook",
]

_MODEL = None
_MODEL_STATUS: dict[str, Any] | None = None

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def load_chess_model(force_reload: bool = False):
    global _MODEL, _MODEL_STATUS
    if _MODEL is not None and not force_reload:
        return _MODEL, CLASSES, _MODEL_STATUS

    if not MODEL_PATH.is_file():
        _MODEL, _MODEL_STATUS = None, {
            "ready": False, "trained": False, "path": str(MODEL_PATH),
            "error": "Model weights are missing. Run the training pipeline first.",
        }
        return None, CLASSES, _MODEL_STATUS

    model = _build_model()
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=True)
        model.eval()
    except Exception as exc:
        logger.exception("Model artifact failed validation")
        _MODEL, _MODEL_STATUS = None, {
            "ready": False, "trained": False, "path": str(MODEL_PATH),
            "error": f"Model artifact is invalid or incompatible: {exc}",
        }
        return None, CLASSES, _MODEL_STATUS

    _MODEL = model
    _MODEL_STATUS = {
        "ready": True, "trained": True, "path": str(MODEL_PATH),
        "classes": len(CLASSES), "architecture": "resnet18",
    }
    return _MODEL, CLASSES, _MODEL_STATUS


def predict_squares(squares_dir: str | Path):
    model, classes, status = load_chess_model()
    if model is None or not status["ready"]:
        raise RuntimeError(status["error"])

    directory = Path(squares_dir)
    paths = [directory / f"square_r{r}_c{c}.jpg" for r in range(8) for c in range(8)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise ValueError(f"Board extraction did not produce all 64 squares: {missing[:3]}")

    images = []
    for path in paths:
        try:
            with Image.open(path) as image:
                images.append(INFERENCE_TRANSFORM(image.convert("RGB")))
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Invalid extracted square: {path}") from exc

    with torch.inference_mode():
        probabilities = torch.softmax(model(torch.stack(images)), dim=1)
        confidences, indices = probabilities.max(dim=1)

    board, confidence_board = [], []
    for row in range(8):
        board.append([])
        confidence_board.append([])
        for col in range(8):
            i = row * 8 + col
            board[row].append(classes[int(indices[i])])
            confidence_board[row].append(round(float(confidences[i]), 4))

    low = sum(c < MIN_CONFIDENCE for row in confidence_board for c in row)
    if low:
        logger.warning("Low-confidence predictions: %d/64", low)
    return board, confidence_board


def reconstruct_board_from_squares(squares_dir: str | Path):
    return predict_squares(squares_dir)[0]
