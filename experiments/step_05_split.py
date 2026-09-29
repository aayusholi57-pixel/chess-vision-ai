import cv2
import numpy as np
import os

# 1. Load, prep, find corners, and warp the board (Full pipeline so far)
image = cv2.imread("board.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
biggest_contour = contours[0]
perimeter = cv2.arcLength(biggest_contour, True)
approx_corners = cv2.approxPolyDP(biggest_contour, 0.02 * perimeter, True)

pts = approx_corners.reshape(4, 2)
rect = np.zeros((4, 2), dtype="float32")
s = pts.sum(axis=1)
rect[0] = pts[np.argmin(s)]
rect[2] = pts[np.argmax(s)]
diff = np.diff(pts, axis=1)
rect[1] = pts[np.argmin(diff)]
rect[3] = pts[np.argmax(diff)]

side_length = 800
dst = np.array([[0, 0], [side_length - 1, 0], [side_length - 1, side_length - 1], [0, side_length - 1]], dtype="float32")
matrix = cv2.getPerspectiveTransform(rect, dst)
warped_board = cv2.warpPerspective(image, matrix, (side_length, side_length))

# 2. Create a folder to save our 64 split squares
output_dir = "experiments/squares"
os.makedirs(output_dir, exist_ok=True)

# 3. Slice the 800x800 board into an 8x8 grid of 100x100 squares
square_size = side_length // 8  # 100 pixels

count = 0
for row in range(8):
    for col in range(8):
        # Calculate pixel boundaries for each square
        y1 = row * square_size
        y2 = (row + 1) * square_size
        x1 = col * square_size
        x2 = (col + 1) * square_size
        
        # Crop the square using NumPy array slicing
        square_image = warped_board[y1:y2, x1:x2]
        
        # Save each square to disk
        file_path = os.path.join(output_dir, f"square_r{row}_c{col}.jpg")
        cv2.imwrite(file_path, square_image)
        count += 1

print(f"Successfully sliced and saved {count} squares into the '{output_dir}/' folder!")

# 4. Display just one sample square (e.g., Top-Left corner) to verify
sample_square = cv2.imread(os.path.join(output_dir, "square_r0_c0.jpg"))
cv2.imshow("Sample Square (Row 0, Col 0)", sample_square)
cv2.waitKey(0)
cv2.destroyAllWindows()