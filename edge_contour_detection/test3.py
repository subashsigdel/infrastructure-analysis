import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load image
img_color = cv2.imread('images/tile_32768_20480.png')
if img_color is None:
    raise FileNotFoundError("Image not found. Check the path.")

img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# STEP 1: Enhance and smooth
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_enhanced = clahe.apply(img_gray)
blur = cv2.GaussianBlur(img_enhanced, (15, 15), 0)

# STEP 2: Adaptive threshold
thresh_adapt_mean = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV,
    181,
    2
)

# STEP 3: Morphological operations
kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
regions = cv2.morphologyEx(thresh_adapt_mean, cv2.MORPH_CLOSE, kernel_large, iterations=3)
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
regions = cv2.morphologyEx(regions, cv2.MORPH_OPEN, kernel_open, iterations=2)

# STEP 4: Filter regions by area
contours_regions, _ = cv2.findContours(regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_region_area = 3000
large_regions = [c for c in contours_regions if cv2.contourArea(c) > min_region_area]

regions_filtered = np.zeros_like(img_gray)
cv2.drawContours(regions_filtered, large_regions, -1, 255, -1)

# STEP 5: Generate one boundary per region and fill with color
img_colored = img_rgb.copy()
boundaries = []
colors = [tuple(np.random.randint(50, 256, 3).tolist()) for _ in range(len(large_regions))]  # brighter colors

for i, region in enumerate(large_regions):
    # Create mask for this single region
    mask = np.zeros_like(img_gray)
    cv2.drawContours(mask, [region], -1, 255, -1)
    
    # Find outer contour for this region
    contours_single, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours_single) > 0:
        c = contours_single[0]
        epsilon = 0.003 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        boundaries.append(approx)
        
        # Fill region with the same color
        cv2.drawContours(img_colored, [approx], -1, colors[i], -1)  # -1 = fill
        
        # Draw boundary slightly darker
        darker = tuple(max(0, x-50) for x in colors[i])
        cv2.drawContours(img_colored, [approx], -1, darker, 2)
        
        # Draw number at centroid
        M = cv2.moments(approx)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(img_colored, str(i+1), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# PLOT RESULTS
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

axes[0].imshow(img_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(regions_filtered, cmap="gray")
axes[1].set_title(f"Detected Regions ({len(large_regions)})")
axes[1].axis("off")

axes[2].imshow(img_colored)
axes[2].set_title(f"Final Boundaries Filled ({len(boundaries)})")
axes[2].axis("off")

plt.tight_layout()
plt.show()
