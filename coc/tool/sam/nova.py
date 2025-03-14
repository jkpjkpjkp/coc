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

import numpy as np
from skimage.morphology import dilation, square

def compute_shared_boundary_length(mask_i, mask_j):
    """
    Compute the length of the shared boundary between two masks using 4-connectivity.
    Counts transitions where mask_i is adjacent to mask_j.
    """
    shared_horizontal = np.sum(mask_i[:,:-1] & mask_j[:,1:])
    shared_vertical = np.sum(mask_i[:-1,:] & mask_j[1:,:])
    return shared_horizontal + shared_vertical

def compute_perimeter(mask):
    """
    Compute the perimeter of a mask by counting transitions from True to False
    or False to True in horizontal and vertical directions (4-connectivity).
    """
    horizontal_trans = np.sum(mask[:,:-1] != mask[:,1:])
    vertical_trans = np.sum(mask[:-1,:] != mask[1:,:])
    return horizontal_trans + vertical_trans

def compute_depth_diffs(mask_i, mask_j, depth_map):
    """
    Compute depth differences across the shared boundary between mask_i and mask_j.
    """
    # Horizontal shared edges
    horizontal_shared = mask_i[:,:-1] & mask_j[:,1:]
    depth_diff_horizontal = np.abs(depth_map[:,:-1] - depth_map[:,1:])[horizontal_shared]

    # Vertical shared edges
    vertical_shared = mask_i[:-1,:] & mask_j[1:,:]
    depth_diff_vertical = np.abs(depth_map[:-1,:] - depth_map[1:,:])[vertical_shared]

    # Combine all depth differences
    depth_diffs = np.concatenate([depth_diff_horizontal, depth_diff_vertical])
    return depth_diffs

def should_merge(mask_i, mask_j, depth_map, overlap_threshold=0.1, depth_threshold=0.05, continuity_threshold=0.8):
    """
    Determine if two masks should be merged based on depth continuity and contour overlap.

    Parameters:
    - mask_i, mask_j: Binary numpy arrays representing the masks.
    - depth_map: Numpy array of depth values with the same shape as the masks.
    - overlap_threshold: Minimum ratio of shared boundary to total perimeter (default: 0.1).
    - depth_threshold: Maximum allowable depth difference for continuity (default: 0.05).
    - continuity_threshold: Minimum proportion of boundary pixels with small depth diffs (default: 0.8).

    Returns:
    - Boolean indicating whether the masks should merge.
    """
    # Check adjacency
    shared_length = compute_shared_boundary_length(mask_i, mask_j)
    if shared_length == 0:
        return False  # Masks are not adjacent

    # Compute perimeters
    perimeter_i = compute_perimeter(mask_i)
    perimeter_j = compute_perimeter(mask_j)
    if perimeter_i == 0 or perimeter_j == 0:
        return False  # Invalid mask

    # Check contour overlap condition
    ratio_i = shared_length / perimeter_i
    ratio_j = shared_length / perimeter_j
    if ratio_i <= overlap_threshold or ratio_j <= overlap_threshold:
        return False  # Shared contour is not a noticeable portion

    # Check depth continuity
    depth_diffs = compute_depth_diffs(mask_i, mask_j, depth_map)
    if len(depth_diffs) == 0:
        return False  # No shared boundary pixels to evaluate
    proportion_continuous = np.sum(depth_diffs < depth_threshold) / len(depth_diffs)
    if proportion_continuous <= continuity_threshold:
        return False  # Depth gradient is not sufficiently continuous

    return True

# Example usage
def merge_masks(masks, depth_map):
    """
    Merge masks based on the defined conditions.

    Parameters:
    - masks: List of binary numpy arrays (masks).
    - depth_map: Depth map as a numpy array.

    Returns:
    - List of merged masks.
    """
    merged = masks.copy()
    n = len(merged)
    merged_flags = [False] * n

    for i in range(n):
        for j in range(i + 1, n):
            if not merged_flags[i] and not merged_flags[j]:
                if should_merge(merged[i]['segmentation'], merged[j]['segmentation'], depth_map):
                    # Merge by taking the union of the masks
                    merged[i]['segmentation'] = merged[i]['segmentation'] | merged[j]['segmentation']
                    merged_flags[j] = True

    # Return only unmerged masks
    return [merged[i]['segmentation'] for i in range(n) if not merged_flags[i]]

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

def depth():
    import cv2
    import torch

    from depth_anything_v2.dpt import DepthAnythingV2

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'/root/autodl-tmp/data/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()

    raw_img = cv2.imread('/root/autodl-tmp/img/cherry/example_0_image_0.png')
    depth = model.infer_image(raw_img, input_size=1024) # HxW raw depth map in numpy
    return depth

def mian():
    from .mask_db import MaskDatabase
    db = MaskDatabase('/root/autodl-tmp/mask_db')
    masks = db.get_list_of_masks('example_0_image_0.png')
    depth_map = depth()
    result = merge_masks(masks, depth_map)
    print("Number of merged masks:", len(result))
    for idx, mask in enumerate(result):
        print(f"Mask {idx}:\n", mask.astype(int))

if __name__ == '__main__':
    mian()