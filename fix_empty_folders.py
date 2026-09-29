import os
import cv2
import numpy as np

print("🔍 Scanning dataset for empty folders...")

# Create a generic 100x100 black square image to use as a placeholder
dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

fixed_count = 0

for split in ["train", "val"]:
    split_dir = os.path.join("dataset", split)
    if not os.path.exists(split_dir): 
        continue
        
    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        if os.path.isdir(class_dir):
            # Check how many files are in this folder
            files = os.listdir(class_dir)
            if len(files) == 0:
                print(f"⚠️ Empty folder found: {class_dir}. Injecting dummy image.")
                cv2.imwrite(os.path.join(class_dir, "dummy_placeholder.jpg"), dummy_image)
                fixed_count += 1

print(f"\n✅ Done! Fixed {fixed_count} empty folders.")