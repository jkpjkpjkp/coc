import pytest
import numpy as np
from coc.tool.sam.mod_server import generate_fn, post_process_mask_data_fn, generate_masks_fn
from tests.tool.test_context import create_dummy_image

@pytest.fixture
def dummy_image():
    return create_dummy_image(300, 300)

def test_mask_generation_pipeline(dummy_image):
    """Test full mask generation pipeline with dummy image"""
    # Convert PIL image to numpy array
    img_np = np.array(dummy_image)

    # Test generate_masks_fn with minimal parameters
    mask_data = generate_masks_fn(
        image=img_np,
        points_per_side=16,
        points_per_batch=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
        stability_score_offset=1.0,
        mask_threshold=0.0,
        box_nms_thresh=0.7,
        crop_n_layers=0,
        crop_nms_thresh=0.7,
        crop_overlap_ratio=0.1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=0,
        use_m2m=False,
        multimask_output=True
    )

    assert isinstance(mask_data, dict)
    assert 'masks' in mask_data
    assert len(mask_data['masks']) > 0

    # Test post-processing
    annotations = post_process_mask_data_fn(
        mask_data_dict=mask_data,
        output_mode="binary_mask"
    )

    assert isinstance(annotations, list)
    assert len(annotations) > 0
    assert 'segmentation' in annotations[0]
    assert 'area' in annotations[0]
    assert 'bbox' in annotations[0]

    # Test full generate_fn integration
    full_annotations = generate_fn(
        image=img_np,
        points_per_side=16,
        points_per_batch=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
        stability_score_offset=1.0,
        mask_threshold=0.0,
        box_nms_thresh=0.7,
        crop_n_layers=0,
        crop_nms_thresh=0.7,
        crop_overlap_ratio=0.1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=0,
        use_m2m=False,
        multimask_output=True,
        output_mode="binary_mask"
    )

    assert isinstance(full_annotations, list)
    assert len(full_annotations) >= len(annotations)                                                                                                                                                                                                                            import pytest
import numpy as np
from coc.tool.sam.mod_server import generate_fn, post_process_mask_data_fn, generate_masks_fn
from tests.tool.test_context import create_dummy_image

def test_sam_mask_generation():
"""Test SAM2 mask generation with dummy image"""
dummy_img = create_dummy_image(300, 300)
img_np = np.array(dummy_img)

# Test basic mask generation
mask_data = generate_masks_fn(
    image=img_np,
    points_per_side=32,
    points_per_batch=64,
    pred_iou_thresh=0.8,
    stability_score_thresh=0.95,
    stability_score_offset=1.0,
    mask_threshold=0.0,
    box_nms_thresh=0.7,
    crop_n_layers=0,
    crop_nms_thresh=0.7,
    crop_overlap_ratio=512/1500,
    crop_n_points_downscale_factor=1,
    min_mask_region_area=0,
    use_m2m=False,
    multimask_output=True
)

assert isinstance(mask_data, dict)
assert 'masks' in mask_data
assert len(mask_data['masks']) > 0

def test_sam_post_processing():
"""Test SAM2 post-processing of mask data"""
dummy_img = create_dummy_image(300, 300)
img_np = np.array(dummy_img)

# First generate mask data
mask_data = generate_masks_fn(
    image=img_np,
    points_per_side=32,
    points_per_batch=64,
    pred_iou_thresh=0.8,
    stability_score_thresh=0.95,
    stability_score_offset=1.0,
    mask_threshold=0.0,
    box_nms_thresh=0.7,
    crop_n_layers=0,
    crop_nms_thresh=0.7,
    crop_overlap_ratio=512/1500,
    crop_n_points_downscale_factor=1,
    min_mask_region_area=0,
    use_m2m=False,
    multimask_output=True
)

# Test post-processing
annotations = post_process_mask_data_fn(
    mask_data_dict=mask_data,
    output_mode="binary_mask"
)

assert isinstance(annotations, list)
assert len(annotations) > 0
assert 'segmentation' in annotations[0]
assert 'area' in annotations[0]
assert 'bbox' in annotations[0]
