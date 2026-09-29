# Model Training Status & Improvement Guide

## Current State
✓ **Model trained**: `models/resnet18_chess.pth` (44.8 MB)
✓ **Training data**: 520 images across 13 classes
✓ **Validation data**: 137 images
✓ **Current accuracy**: ~25% (validation set)

### Class Distribution Issues
Classes with too few samples:
- `black_king`: 1 train, 1 val (needs 50+)
- `empty`: 1 train, 1 val (needs 50+)
- `black_pawn`: 29 train (needs 100+)
- `white_pawn`: 36 train (needs 100+)

## To Improve Accuracy

### Option 1: Collect More Data (Recommended)
- Collect 100-200 labeled images per piece type
- Vary: lighting, angles, board backgrounds, piece styles
- Use data augmentation (rotation, brightness, crops)

### Option 2: Synthetic Data Generation
- Use chess board rendering libraries (python-chess, PIL)
- Generate thousands of synthetic board configurations
- Render with different styles/backgrounds

### Option 3: Use Pre-trained Model
- Find a chess piece classifier on Hugging Face/GitHub
- Fine-tune on your dataset instead of training from scratch

### Option 4: Ensemble Approach
- Train multiple models on different data subsets
- Combine predictions for better accuracy

## How to Retrain After Collecting More Data

1. Place images in `dataset/train/<class_name>/` and `dataset/val/<class_name>/`
2. Run: `cd training && python train.py`
3. The new model overwrites `models/resnet18_chess.pth`

## Current Training Script
- **Framework**: PyTorch ResNet-18 with transfer learning
- **Data augmentation**: Rotation, horizontal flip, color jitter
- **Epochs**: 10
- **Batch size**: 32
- **Optimizer**: Adam (lr=0.001)
- **Best loss saved**: `models/resnet18_chess_best.pth`

## Monitoring
Check training progress in logs. Early stopping could help prevent overfitting.

---
For production use with current 25% accuracy:
- Use ensemble voting from multiple inferences
- Validate predictions with chess rule checking (already implemented in `app/chess_engine.py`)
- Flag low-confidence predictions for manual review
