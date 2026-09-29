import cv2
import numpy as np

# 1. Load and prep the image (Same as before)
image = cv2.imread("board.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

# 2. Find all the shapes (contours) in the edge map
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 3. Sort the shapes by size, largest first, and grab the biggest one
contours = sorted(contours, key=cv2.contourArea, reverse=True)
biggest_contour = contours[0]

# 4. Approximate that messy hand-drawn shape into a perfect polygon
perimeter = cv2.arcLength(biggest_contour, True)
approx_corners = cv2.approxPolyDP(biggest_contour, 0.02 * perimeter, True)

# 5. Draw a thick green line around the detected board
cv2.drawContours(image, [approx_corners], -1, (0, 255, 0), 5)

cv2.imshow("Detected Chessboard", image)
cv2.waitKey(0)
cv2.destroyAllWindows()