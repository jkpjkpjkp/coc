"""visual to depth.

currently cutting is right but to-depth alg is wrong.
my depth result is in this particular format: a single image, left side original, rightside depth, red is closer blue further. the middle is separated wish purely white (exact center, unknown width). so i need you to 0. read in this png 1. determine how much pure white is in the exact center (maximal width of exact which on exact center is the padd) 3. cut out the depth heatmap on the right 4. re-calculate depth based on color. you start coding and i go see if i have exact mapping of color->depth.
"""

import cv2
import numpy as np

# Read the PNG image
img = cv2.imread('cherry_up/example_0_image_0.png')

def is_white_column(img, x):
    """Check if all pixels in column x are white ([255, 255, 255] in BGR)."""
    return np.all(img[:, x] == [255, 255, 255], axis=1).all()

# Get image dimensions
H, W = img.shape[:2]
center = W // 2

# Find the left boundary
left = center
while left >= 0 and is_white_column(img, left):
    left -= 1
start_white = left + 1  # First white column

# Find the right boundary
right = center
while right < W and is_white_column(img, right):
    right += 1
end_white = right - 1  # Last white column

# Calculate the width of the white strip
white_width = end_white - start_white + 1
print(f"Width of the central white strip: {white_width} pixels")


# Extract the depth map
depth_map = img[:, end_white + 1:]

# Save the depth map as a new image

cv2.imwrite('depth_map.png', depth_map)


# Convert depth map to HSV
hsv = cv2.cvtColor(depth_map, cv2.COLOR_BGR2HSV)
hue = hsv[:, :, 0]  # Hue channel (0 to 179)

# Recalculate depth (assuming hue 0 to 120 maps to depth 0 to max_depth)
# For now, normalize depth to [0, 1] assuming hue range 0 to 120
depth = hue.astype(float) / 180.0

# Optional: Check actual hue range for validation
min_hue, max_hue = np.min(hue), np.max(hue)
print(f"Hue range in depth map: {min_hue} to {max_hue}")

# Optional: Save depth as a grayscale image for visualization
depth_vis = (depth * 255).astype(np.uint8)
cv2.imwrite('depth.png', depth_vis)