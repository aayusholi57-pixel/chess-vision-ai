import os
import shutil
import glob
import random
import kagglehub

print("🚀 1. Downloading Kaggle dataset...")
kaggle_path = kagglehub.dataset_download("anshulmehtakaggl/chess-pieces-detection-images-dataset")
print(f"Dataset downloaded to: {kaggle_path}")

classes = [
    "empty", "white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king",
    "black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king"
]

print("🧹 2. Wiping old dataset and creating fresh folders...")
if os.path.exists("dataset"):
    shutil.rmtree("dataset")
for split in ["train", "val"]:
    for cls in classes:
        os.makedirs(os.path.join("dataset", split, cls), exist_ok=True)

print("🔍 3. Scanning downloaded images...")
image_files = []
for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
    image_files.extend(glob.glob(os.path.join(kaggle_path, '**', ext), recursive=True))

def determine_class(filepath):
    """Smart sorter: reads the file path AND folder names to guess the piece and color."""
    filepath_lower = filepath.lower()
    
    # 1. Check for empty
    if "empty" in filepath_lower or "blank" in filepath_lower:
        return "empty"
        
    # 2. Extract piece type from the path
    piece = None
    if "pawn" in filepath_lower: piece = "pawn"
    elif "knight" in filepath_lower: piece = "knight"
    elif "bishop" in filepath_lower: piece = "bishop"
    elif "rook" in filepath_lower: piece = "rook"
    elif "queen" in filepath_lower: piece = "queen"
    elif "king" in filepath_lower: piece = "king"
    
    # 3. Guess color based on the dataset structure or image content. 
    # NOTE: Since this dataset doesn't label color, we will randomly assign color for now 
    # to test the pipeline. A real-world dataset MUST have explicit black/white labels.
    if piece:
        color = random.choice(["white", "black"])
        return f"{color}_{piece}"
    
    return None

print("🔀 4. Sorting and splitting into Train (80%) and Val (20%)...")
class_to_files = {cls: [] for cls in classes}
skipped_count = 0

for img_path in image_files:
    detected_class = determine_class(img_path)
    if detected_class:
        class_to_files[detected_class].append(img_path)
    else:
        skipped_count += 1

copied_count = 0
for cls, files in class_to_files.items():
    if not files:
        continue
    random.shuffle(files)
    split_index = int(len(files) * 0.8)
    
    train_files = files[:split_index]
    val_files = files[split_index:]
    
    for f in train_files:
        shutil.copy(f, os.path.join("dataset", "train", cls, os.path.basename(f)))
        copied_count += 1
    for f in val_files:
        shutil.copy(f, os.path.join("dataset", "val", cls, os.path.basename(f)))
        copied_count += 1

print(f"\n✅ Success! Copied {copied_count} images into your 'dataset/' folder.")
if skipped_count > 0:
    print(f"⚠️ Warning: Skipped {skipped_count} images.")