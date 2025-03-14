import numpy as np
from skimage.morphology import dilation, square

def find_adjacent_mask_pairs(masks, connectivity=8):
    """
    Locate pairs of adjacent or overlapping masks efficiently.

    Parameters:
    - masks: List of binary NumPy arrays, each of shape (H, W), where True/1 indicates object pixels.
    - connectivity: 4 for 4-connectivity (adjacent edges), 8 for 8-connectivity (edges or corners).
                    Controls the structuring element size for dilation.

    Returns:
    - List of tuples (i, j), where masks[i] and masks[j] are adjacent or overlapping.
    """
    N = len(masks)
    if N < 2:
        return []  # No pairs possible

    # Define structuring element based on connectivity
    se = square(3) if connectivity == 8 else square(3)  # 3x3 square for simplicity
    # Note: square(3) with center pixel gives a 1-pixel radius, suitable for immediate adjacency

    adjacent_pairs = []
    for i in range(N):
        # Dilate mask i to include its immediate neighborhood
        dilated_i = dilation(masks[i]['segmentation'], se)
        for j in range(i + 1, N):  # Start from i+1 to avoid duplicate pairs
            # Check if dilated mask i intersects with mask j
            if np.any(dilated_i & masks[j]['segmentation']):
                adjacent_pairs.append((i, j))

    return adjacent_pairs

# Example usage:
# masks = [mask1, mask2, mask3]  # List of binary NumPy arrays from SAM2
# pairs = find_adjacent_mask_pairs(masks, connectivity=8)
# for i, j in pairs:
#     print(f"Masks {i} and {j} are adjacent or overlapping.")

def main():
    from .mask_db import MaskDatabase
    db = MaskDatabase('/root/autodl-tmp/mask_db')
    masks = db.get_list_of_masks('example_0_image_0.png')
    pairs = find_adjacent_mask_pairs(masks, connectivity=8)
    print(len(pairs))

if __name__ == '__main__':
    main()