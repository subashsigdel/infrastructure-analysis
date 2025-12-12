import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load image
img_color = cv2.imread('images/tile_82944_46080.png')
if img_color is None:
    raise FileNotFoundError("Image not found. Check the path.")

# Convert BGR to RGB for plotting
img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# STEP 1: ENHANCE AND SMOOTH
# Enhance contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_enhanced = clahe.apply(img_gray)

# Heavy blur to remove internal texture
blur = cv2.GaussianBlur(img_enhanced, (15, 15), 0)

# STEP 2: ADAPTIVE THRESHOLDING TO FIND REGIONS
thresh_adapt_mean = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV,
    181,  # Large blockSize ignores internal patterns
    2
)

# STEP 3: MORPHOLOGICAL OPERATIONS TO CLEAN REGIONS
# Close to merge same-field areas
kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
regions = cv2.morphologyEx(thresh_adapt_mean, cv2.MORPH_CLOSE, kernel_large, iterations=3)

# Open to remove small noise
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
regions = cv2.morphologyEx(regions, cv2.MORPH_OPEN, kernel_open, iterations=2)

# STEP 4: FILTER REGIONS BY SIZE
# Only keep large field-sized regions
contours_regions, _ = cv2.findContours(regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_region_area = 3000  # Minimum field size
large_regions = [c for c in contours_regions if cv2.contourArea(c) > min_region_area]

# Create mask with only large regions
regions_filtered = np.zeros_like(img_gray)
cv2.drawContours(regions_filtered, large_regions, -1, 255, -1)

# STEP 5: DETECT BOUNDARIES ONLY AT REGION EDGES
# This is the KEY - we only look for boundaries at region perimeters
# Detect edges on the region mask (not the original image!)
edges = cv2.Canny(regions_filtered, 50, 150)

# Connect fragmented edges with directional kernels
kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11))
edges_h = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_h)
edges_v = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_v)
edges_connected = cv2.bitwise_or(edges_h, edges_v)

# Additional closing
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
edges_connected = cv2.morphologyEx(edges_connected, cv2.MORPH_CLOSE, kernel_close, iterations=2)

# Dilate to strengthen boundaries
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
edges_connected = cv2.dilate(edges_connected, kernel_dilate, iterations=1)

# STEP 6: CRITICAL FILTER - Remove internal boundaries
# Erode regions to get field centers, remove any boundary inside
kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
field_interiors = cv2.erode(regions_filtered, kernel_erode, iterations=1)

# Remove any boundary that's inside a field
edges_external_only = edges_connected.copy()
edges_external_only[field_interiors > 0] = 0

# STEP 7: EXTRACT AND FILTER CONTOURS
contours, _ = cv2.findContours(edges_external_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter by perimeter (boundaries should be reasonably long)
min_perimeter = 80
filtered_contours = []

for c in contours:
    perimeter = cv2.arcLength(c, True)
    area = cv2.contourArea(c)
    
    if perimeter >= min_perimeter:
        # Additional shape filter
        if perimeter > 0:
            # Compactness check
            compactness = 4 * np.pi * area / (perimeter * perimeter)
            
            # Keep boundaries (low compactness = elongated)
            if compactness < 0.5:  # Not too circular/blob-like
                # Smooth the contour
                epsilon = 0.003 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
                filtered_contours.append(approx)

# STEP 8: CREATE FINAL BOUNDARY MASK
edges_clean = np.zeros_like(edges_connected)
cv2.drawContours(edges_clean, filtered_contours, -1, 255, 2)

# VISUALIZATIONS

# 1. Red overlay on original
img_with_edges = img_rgb.copy()
edge_mask = edges_clean > 0
img_with_edges[edge_mask] = [255, 0, 0]

# 2. Semi-transparent overlay
img_overlay = img_rgb.copy()
overlay = np.zeros_like(img_rgb)
overlay[edge_mask] = [255, 0, 0]
img_overlay = cv2.addWeighted(img_rgb, 0.85, overlay, 2, 0)

# 3. Show regions with boundaries
img_regions = img_rgb.copy()
regions_colored = np.zeros_like(img_rgb)
regions_colored[regions_filtered > 0] = [0, 80, 0]  # Green tint for regions
img_regions = cv2.addWeighted(img_rgb, 0.8, regions_colored, 0.2, 0)
img_regions[edge_mask] = [255, 0, 0]

# 4. Show field interiors (debugging)
img_debug = img_rgb.copy()
interior_colored = np.zeros_like(img_rgb)
interior_colored[field_interiors > 0] = [0, 0, 150]  # Blue = interior (boundaries removed here)
img_debug = cv2.addWeighted(img_rgb, 0.7, interior_colored, 0.3, 0)
img_debug[edge_mask] = [255, 0, 0]

# 5. Comparison: before and after interior removal
img_before = img_rgb.copy()
img_before[edges_connected > 0] = [255, 255, 0]  # Yellow = all boundaries before filtering

# PLOT RESULTS

fig, axes = plt.subplots(2, 4, figsize=(24, 12))

# Row 1: Processing pipeline
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("Original Image", fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(blur, cmap='gray')
axes[0, 1].set_title(f"Heavy Blur (25×25)\nRemoves internal texture", fontsize=11)
axes[0, 1].axis('off')

axes[0, 2].imshow(regions_filtered, cmap='gray')
axes[0, 2].set_title(f"Detected Regions\n({len(large_regions)} fields, min area={min_region_area})", fontsize=11)
axes[0, 2].axis('off')

axes[0, 3].imshow(field_interiors, cmap='gray')
axes[0, 3].set_title("Field Interiors\n(Boundaries here will be removed)", fontsize=11)
axes[0, 3].axis('off')

# Row 2: Boundary results
axes[1, 0].imshow(img_before)
axes[1, 0].set_title("Before Filter\n(Yellow = all detected boundaries)", fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(edges_clean, cmap='gray')
axes[1, 1].set_title(f"After Filter\n({len(filtered_contours)} boundaries kept)", fontsize=11, fontweight='bold')
axes[1, 1].axis('off')

axes[1, 2].imshow(img_overlay)
axes[1, 2].set_title("Subtle Overlay on Original", fontsize=11)
axes[1, 2].axis('off')

axes[1, 3].imshow(img_with_edges)
axes[1, 3].set_title("FINAL RESULT", fontsize=12, fontweight='bold')
axes[1, 3].axis('off')

plt.tight_layout()
plt.show()

# STATISTICS

print("\n" + "="*70)
print("FIELD BOUNDARY DETECTION RESULTS")
print("="*70)
print(f"Image size: {img_gray.shape[0]} × {img_gray.shape[1]} pixels")
print(f"Detected field regions: {len(large_regions)}")
print(f"Total boundaries before filter: {len(contours)}")
print(f"Final boundaries after filter: {len(filtered_contours)}")
print(f"Boundaries removed: {len(contours) - len(filtered_contours)}")

if large_regions:
    areas = [cv2.contourArea(c) for c in large_regions]
    print(f"\nField Size Statistics:")
    print(f"  Average: {np.mean(areas):.0f} pixels²")
    print(f"  Largest: {np.max(areas):.0f} pixels²")
    print(f"  Smallest: {np.min(areas):.0f} pixels²")

if filtered_contours:
    perimeters = [cv2.arcLength(c, True) for c in filtered_contours]
    print(f"\nBoundary Length Statistics:")
    print(f"  Total length: {sum(perimeters):.0f} pixels")
    print(f"  Average: {np.mean(perimeters):.0f} pixels")
    print(f"  Longest: {max(perimeters):.0f} pixels")
