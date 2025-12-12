import cv2
import matplotlib.pyplot as plt

# Load image
img_color = cv2.imread('images/tile_82944_46080.png')
if img_color is None:
    raise FileNotFoundError("Image not found. Check the path.")

# Convert BGR to RGB for plotting
img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

# Convert to grayscale
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur = cv2.GaussianBlur(img_gray, (7,7), 0)

# Simple Thresholding
_, thresh_simple = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY_INV)

# Otsu Thresholding
_, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Adaptive Mean Thresholding
# Using your values: blockSize=21, C=10
thresh_adapt_mean = cv2.adaptiveThreshold(
    blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 2
)

# Adaptive Gaussian Thresholding
# Using your values: blockSize=21, C=4
thresh_adapt_gauss = cv2.adaptiveThreshold(
    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 2
)

# Morphological filtering to clean up masks
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
simple_clean = cv2.morphologyEx(thresh_simple, cv2.MORPH_CLOSE, kernel)
otsu_clean = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)
mean_clean = cv2.morphologyEx(thresh_adapt_mean, cv2.MORPH_CLOSE, kernel)
gauss_clean = cv2.morphologyEx(thresh_adapt_gauss, cv2.MORPH_CLOSE, kernel)

# Find contours from Gaussian adaptive threshold (example)
contours, _ = cv2.findContours(gauss_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_area = 3000
filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

# Draw contours on original image
img_with_contours = img_rgb.copy()
cv2.drawContours(img_with_contours, filtered_contours, -1, (255,0,0), 2)

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(24,12))

axes[0,0].imshow(img_rgb)
axes[0,0].set_title("Original Image")
axes[0,0].axis('off')

axes[0,1].imshow(simple_clean, cmap='gray')
axes[0,1].set_title("Simple Threshold + Morphology")
axes[0,1].axis('off')

axes[0,2].imshow(otsu_clean, cmap='gray')
axes[0,2].set_title("Otsu Threshold + Morphology")
axes[0,2].axis('off')

axes[1,0].imshow(mean_clean, cmap='gray')
axes[1,0].set_title("Adaptive Mean + Morphology")
axes[1,0].axis('off')

axes[1,1].imshow(gauss_clean, cmap='gray')
axes[1,1].set_title("Adaptive Gaussian + Morphology")
axes[1,1].axis('off')

axes[1,2].imshow(img_with_contours)
axes[1,2].set_title("Filtered Contours (Gaussian Adaptive)")
axes[1,2].axis('off')

plt.tight_layout()
plt.show()
