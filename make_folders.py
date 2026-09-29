import os

# Define the 13 chess classes
classes = [
    "empty",
    "white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king",
    "black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king"
]

# Create them inside dataset/train and dataset/val
for split in ["train", "val"]:
    for cls in classes:
        dir_path = os.path.join("dataset", split, cls)
        os.makedirs(dir_path, exist_ok=True)

print("Successfully created 13 class folders for train and val using Python!")