import cv2

# 1. Load the image from your hard drive into the computer's memory
image = cv2.imread("board.png")

# 2. Open a window and show the image on the screen
cv2.imshow("Chess Vision AI - Step 1", image)

# 3. Tell the computer to pause and wait for you to press a key
cv2.waitKey(0)

# 4. Clean up and close the window safely
cv2.destroyAllWindows()