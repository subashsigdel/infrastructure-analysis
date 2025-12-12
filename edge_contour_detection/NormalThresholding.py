import cv2
import matplotlib.pyplot as plt

# Load image
img_color = cv2.imread('images/tile_27648_25600_png.rf.6027a20b8530056b286eaba0f2381753.jpg')
if img_color is None:
    raise FileNotFoundError("Image not found. Check the path.")

# Convert BGR to RGB for matplotlib
img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

# Convert to grayscale
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Threshold to get binary image
_, threshold = cv2.threshold(img_gray, 140, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out small contours
min_area = 1000  # minimum area to keep
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# Draw filtered contours on a copy of the original image
img_with_contours = img_rgb.copy()
cv2.drawContours(img_with_contours, filtered_contours, -1, (255, 0, 0), 2)  # red contours

# Plot images
fig, axes = plt.subplots(1, 4, figsize=(24, 6))

axes[0].imshow(img_rgb)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(img_gray, cmap='gray')
axes[1].set_title("Grayscale Image")
axes[1].axis('off')

axes[2].imshow(threshold, cmap='gray')
axes[2].set_title("Thresholded Image")
axes[2].axis('off')

axes[3].imshow(img_with_contours)
axes[3].set_title("Filtered Contours (Large Areas Only)")
axes[3].axis('off')

plt.tight_layout()
plt.show()
