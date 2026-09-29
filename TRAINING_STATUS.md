# Training Status

The previous documentation claimed a trained model and accuracy while the model artifact was not present on GitHub. That claim has been removed.

Run:

    python -m training.train --epochs 15
    python -m training.evaluate

Generated artifacts:

- models/resnet18_chess_best.pth
- models/resnet18_chess.pth
- models/training_history.json
- models/evaluation.json

Training uses weighted sampling because the included dataset is class-imbalanced.

For repeatable cloud training, use GitHub Actions -> Train Chess Model. The workflow stores the trained model and evaluation files as artifacts.

A production accuracy claim should only be made after inspecting held-out metrics and testing photographs from piece sets that were not used for training.
