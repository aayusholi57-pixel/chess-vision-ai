import cv2
import numpy as np
import os

def process_uploaded_image(image_path, output_dir="app/temp_squares"):
    """
    Reads an uploaded image, flattens the chessboard, and slices it into 64 squares.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load image and find edges
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image file.")
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # 2. Find the board corners
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No chessboard detected.")
        
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    perimeter = cv2.arcLength(contours[0], True)
    approx_corners = cv2.approxPolyDP(contours[0], 0.02 * perimeter, True)

    if len(approx_corners) != 4:
        # Fallback: If we can't find a perfect board, just slice the original image
        warped_board = cv2.resize(image, (800, 800))
    else:
        # 3. Flatten the board (Perspective Warp)
        pts = approx_corners.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        dst = np.array([[0, 0], [799, 0], [799, 799], [0, 799]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped_board = cv2.warpPerspective(image, matrix, (800, 800))

    # 4. Slice into 64 squares
    square_size = 100
    for row in range(8):
        for col in range(8):
            y1, y2 = row * square_size, (row + 1) * square_size
            x1, x2 = col * square_size, (col + 1) * square_size
            square = warped_board[y1:y2, x1:x2]
            
            # Save to temporary folder
            cv2.imwrite(os.path.join(output_dir, f"square_r{row}_c{col}.jpg"), square)

    return output_dir