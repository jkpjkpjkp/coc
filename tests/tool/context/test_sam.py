import pytest
import numpy as np
from PIL import Image
from coc.tool.context import sam_predict, sam_auto  # Replace 'your_module' with the actual module name

# Fixture to load sample images
@pytest.fixture
def image(request):
    """Load an image based on the parametrized image name."""
    image_name = request.param
    image_path = f"data/sample/{image_name}.jpg"
    return Image.open(image_path)

# Tests for sam_predict
@pytest.mark.parametrize("image", ["onions", "4girls"], indirect=True)
def test_sam_predict_point(image):
    """Test sam_predict with a single point prompt and multimask_output=False."""
    W, H = image.size
    point_coords = np.array([[W // 2, H // 2]])  # Center of the image
    point_labels = np.array([1])  # Foreground point
    masks, scores, low_res_logits = sam_predict(
        image,
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
        normalize_coords=False
    )
    # Check types
    assert isinstance(masks, np.ndarray), "Masks should be a NumPy array"
    assert isinstance(scores, np.ndarray), "Scores should be a NumPy array"
    assert isinstance(low_res_logits, np.ndarray), "Low-res logits should be a NumPy array"
    # Check shapes
    assert masks.shape == (1, H, W), f"Expected masks shape (1, {H}, {W}), got {masks.shape}"
    assert scores.shape == (1,), f"Expected scores shape (1,), got {scores.shape}"
    assert low_res_logits.shape == (1, 256, 256), f"Expected low_res_logits shape (1, 256, 256), got {low_res_logits.shape}"
    # Check mask is binary (since return_logits=False by default)
    assert masks.dtype == bool or (masks.dtype == np.float32 and np.all(np.isin(masks, [0, 1]))), "Masks should be binary"

@pytest.mark.parametrize("image", ["onions", "4girls"], indirect=True)
def test_sam_predict_box(image):
    """Test sam_predict with a box prompt and multimask_output=False."""
    W, H = image.size
    box = np.array([W // 4, H // 4, 3 * W // 4, 3 * H // 4])  # Central box in XYXY format
    masks, scores, low_res_logits = sam_predict(
        image,
        box=box,
        multimask_output=False
    )
    # Check types
    assert isinstance(masks, np.ndarray), "Masks should be a NumPy array"
    assert isinstance(scores, np.ndarray), "Scores should be a NumPy array"
    assert isinstance(low_res_logits, np.ndarray), "Low-res logits should be a NumPy array"
    # Check shapes
    assert masks.shape == (1, H, W), f"Expected masks shape (1, {H}, {W}), got {masks.shape}"
    assert scores.shape == (1,), f"Expected scores shape (1,), got {scores.shape}"
    assert low_res_logits.shape == (1, 256, 256), f"Expected low_res_logits shape (1, 256, 256), got {low_res_logits.shape}"
    # Check mask is binary
    assert masks.dtype == bool or (masks.dtype == np.float32 and np.all(np.isin(masks, [0, 1]))), "Masks should be binary"

@pytest.mark.parametrize("image", ["onions", "4girls"], indirect=True)
def test_sam_predict_multimask(image):
    """Test sam_predict with a point prompt and multimask_output=True."""
    W, H = image.size
    point_coords = np.array([[W // 2, H // 2]])  # Center of the image
    point_labels = np.array([1])  # Foreground point
    masks, scores, low_res_logits = sam_predict(
        image,
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
        normalize_coords=False
    )
    # Check types
    assert isinstance(masks, np.ndarray), "Masks should be a NumPy array"
    assert isinstance(scores, np.ndarray), "Scores should be a NumPy array"
    assert isinstance(low_res_logits, np.ndarray), "Low-res logits should be a NumPy array"
    # Check shapes (expect 3 masks when multimask_output=True)
    assert masks.shape == (3, H, W), f"Expected masks shape (3, {H}, {W}), got {masks.shape}"
    assert scores.shape == (3,), f"Expected scores shape (3,), got {scores.shape}"
    assert low_res_logits.shape == (3, 256, 256), f"Expected low_res_logits shape (3, 256, 256), got {low_res_logits.shape}"
    # Check mask is binary
    assert masks.dtype == bool or (masks.dtype == np.float32 and np.all(np.isin(masks, [0, 1]))), "Masks should be binary"

# Tests for sam_auto
@pytest.mark.parametrize("image", ["onions", "4girls"], indirect=True)
def test_sam_auto_default(image):
    """Test sam_auto with default parameters."""
    results = sam_auto(image)
    # Check output is a non-empty list of dictionaries
    assert isinstance(results, list), "Results should be a list"
    assert len(results) > 0, "Results should contain at least one mask"
    for result in results:
        assert isinstance(result, dict), "Each result should be a dictionary"
        # Check required keys and their types
        assert 'segmentation' in result, "'segmentation' key missing"
        assert isinstance(result['segmentation'], np.ndarray), "Segmentation should be a NumPy array"
        assert result['segmentation'].shape == (image.height, image.width), f"Expected segmentation shape ({image.height}, {image.width}), got {result['segmentation'].shape}"
        assert result['segmentation'].dtype == bool or (result['segmentation'].dtype == np.float32 and np.all(np.isin(result['segmentation'], [0, 1]))), "Segmentation should be binary"

        assert 'bbox' in result, "'bbox' key missing"
        assert isinstance(result['bbox'], list), "Bbox should be a list"
        assert len(result['bbox']) == 4, "Bbox should have 4 elements"
        assert all(isinstance(x, (int, float)) for x in result['bbox']), "Bbox elements should be numbers"

        assert 'area' in result, "'area' key missing"
        assert isinstance(result['area'], int), "Area should be an integer"

        assert 'predicted_iou' in result, "'predicted_iou' key missing"
        assert isinstance(result['predicted_iou'], float), "Predicted IoU should be a float"

        assert 'point_coords' in result, "'point_coords' key missing"
        assert isinstance(result['point_coords'], list), "Point coords should be a list"

        assert 'stability_score' in result, "'stability_score' key missing"
        assert isinstance(result['stability_score'], float), "Stability score should be a float"

        assert 'crop_box' in result, "'crop_box' key missing"
        assert isinstance(result['crop_box'], list), "Crop box should be a list"
        assert len(result['crop_box']) == 4, "Crop box should have 4 elements"