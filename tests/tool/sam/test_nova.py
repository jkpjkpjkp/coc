import numpy as np
from coc.tool.sam.nova import find_adjacent_mask_pairs, merge_masks

def test_merge_masks():
    # Test Case 1: Should merge
    mask1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)
    mask2 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
    masks = [{'segmentation': mask1}, {'segmentation': mask2}]
    depth_map = np.array([[0., 0., 0.], [0., 1.0, 1.01], [0., 0., 0.]], dtype=float)

    result = merge_masks(masks, depth_map)
    print("Test 1 - Should Merge:")
    print("Number of masks:", len(result))
    expected = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]], dtype=bool)
    print("Merged mask:\n", result[0].astype(int))
    print("Expected:\n", expected.astype(int))
    print("Pass:", np.array_equal(result[0], expected) and len(result) == 1)

    # Test Case 2: Shouldn't merge
    depth_map = np.array([[0., 0., 0.], [0., 1.0, 2.0], [0., 0., 0.]], dtype=float)
    result = merge_masks(masks, depth_map)
    print("\nTest 2 - Shouldn't Merge:")
    print("Number of masks:", len(result))
    print("Mask 0:\n", result[0].astype(int))
    print("Mask 1:\n", result[1].astype(int))
    print("Pass:", len(result) == 2 and np.array_equal(result[0], mask1) and np.array_equal(result[1], mask2))

if __name__ == '__main__':
    test_merge_masks()
    # mian()  # Uncomment to run original main if needed