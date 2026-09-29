import cv2
import numpy as np

# 1. Load, prep, and find the 4 corners of the board (from M1 & M2)
image = cv2.imread("board.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
biggest_contour = contours[0]
perimeter = cv2.arcLength(biggest_contour, True)
approx_corners = cv2.approxPolyDP(biggest_contour, 0.02 * perimeter, True)

# Ensure we actually found 4 corners
if len(approx_corners) == 4:
    # Reshape the corners into a clean 4x2 matrix and sort them 
    # (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    pts = approx_corners.reshape(4, 2)
    
    # Mathematical sorting based on coordinates
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]        # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)]        # Bottom-right has largest sum
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]     # Top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]     # Bottom-left has largest difference

    # 2. Define the size of our clean, flattened destination image (e.g., 800x800 pixels)
    side_length = 800
    dst = np.array([
        [0, 0],
        [side_length - 1, 0],
        [side_length - 1, side_length - 1],
        [0, side_length - 1]
    ], dtype="float32")

    # 3. Calculate the mathematical Transformation Matrix
    matrix = cv2.getPerspectiveTransform(rect, dst)

    # 4. Apply the transformation to warp the image flat!
    warped_board = cv2.warpPerspective(image, matrix, (side_length, side_length))

    # 5. Show the magic result
    cv2.imshow("Original Tilted Board", image)
    cv2.imshow("Flattened Birds-Eye Board", warped_board)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Could not find 4 distinct corners. Try a clearer image.")