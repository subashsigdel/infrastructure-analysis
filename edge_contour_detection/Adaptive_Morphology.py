import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load image
img_color = cv2.imread('images/tile_82944_46080.png)
if img_color is None:
    raise FileNotFoundError("Image not found. Check the path.")

# Convert BGR to RGB for plotting
img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

# Convert to grayscale
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur = cv2.GaussianBlur(img_gray, (7,7), 0)

# Adaptive Mean Thresholding
thresh_adapt_mean = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV,
    51,  # blockSize
    2    # C value
)

# Morphological filtering to clean up masks
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
mean_clean = cv2.morphologyEx(thresh_adapt_mean, cv2.MORPH_CLOSE, kernel)

# Detect edges using Canny
edges = cv2.Canny(mean_clean, 50, 150)

# Connect fragmented edges
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
edges_connected = cv2.dilate(edges, kernel_dilate, iterations=1)
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
edges_connected = cv2.morphologyEx(edges_connected, cv2.MORPH_CLOSE, kernel_close)

# Remove small blobs using contours
# Find contours from connected edges
contours, _ = cv2.findContours(edges_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_area = 2500
filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

# Create clean edge mask from filtered contours
edges_clean = np.zeros_like(edges_connected)
cv2.drawContours(edges_clean, filtered_contours, -1, 255, 2)

# Overlay filtered edges in red on original image
img_with_edges = img_rgb.copy()
edge_mask = edges_clean != 0
img_with_edges[edge_mask] = [255, 0, 0]  # red color

# Plot results
fig, axes = plt.subplots(1, 4, figsize=(24,6))

axes[0].imshow(img_rgb)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(img_gray, cmap='gray')
axes[1].set_title("Grayscale Image")
axes[1].axis('off')

axes[2].imshow(mean_clean, cmap='gray')
axes[2].set_title("Adaptive Mean + Morphology")
axes[2].axis('off')

axes[3].imshow(img_with_edges)
axes[3].set_title("Filtered Edges Overlay (Red)")
axes[3].axis('off')

plt.tight_layout()
plt.show()
