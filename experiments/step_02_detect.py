import cv2

# 1. Load the image
image = cv2.imread("board.png")

# 2. Convert the image from Color to Grayscale (Black and White)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Blur the image slightly to remove tiny details (like wood grain)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 4. Find the edges! (Canny Edge Detection)
edges = cv2.Canny(blurred, 50, 150)

# 5. Show the original and the new "edges" view side-by-side
cv2.imshow("Original Image", image)
cv2.imshow("Computer Vision - Edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()