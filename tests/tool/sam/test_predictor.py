import pytest
import numpy as np
from PIL import Image
from coc.tool.sam.automagen import SAMWrapper, get_sam  # Adjust import path
from coc.tool.sam.predictor import SAM2PredictorWrapper, get_sam_predictor  # Adjust import path

# Fixture for sample image
@pytest.fixture
def sample_image():
    """Provides a sample image from data/sample/onions.jpg."""
    return Image.open('data/sample/onions.jpg')

# --- Tests for SAMWrapper ---
def test_sam_wrapper_loading():
    """Test that SAMWrapper initializes correctly."""
    wrapper = SAMWrapper(variant='t', max_parallel=1)  # 't' for tiny variant
    assert wrapper.model is not None, "Model should be loaded"

def test_sam_wrapper_generate_masks(sample_image):
    """Test mask generation with a sample image."""
    wrapper = SAMWrapper(variant='t', max_parallel=1)
    mask_data = wrapper.generate_masks(sample_image)
    assert isinstance(mask_data, dict), "Mask data should be a dictionary"
    assert 'masks' in mask_data, "Mask data should contain 'masks' key"
    assert isinstance(mask_data['masks'], np.ndarray), "Masks should be a NumPy array"
    assert mask_data['masks'].shape[0] > 0, "At least one mask should be generated"
    assert mask_data['masks'].shape[1:] == sample_image.shape[:2], "Mask dimensions should match image"

def test_sam_wrapper_post_process_masks(sample_image):
    """Test post-processing of generated masks."""
    wrapper = SAMWrapper(variant='t', max_parallel=1)
    mask_data = wrapper.generate_masks(sample_image)
    annotations = wrapper.post_process_masks(mask_data)
    assert isinstance(annotations, list), "Annotations should be a list"
    assert len(annotations) > 0, "At least one annotation should be produced"
    assert 'segmentation' in annotations[0], "Each annotation should have a 'segmentation' key"
    assert isinstance(annotations[0]['segmentation'], np.ndarray), "Segmentation should be a NumPy array"

def test_sam_wrapper_visualize_annotations(sample_image):
    """Test visualization of annotations."""
    wrapper = SAMWrapper(variant='t', max_parallel=1)
    annotations = wrapper._run(sample_image)  # Assuming _run is the internal method
    vis_image = wrapper.visualize_annotations(sample_image, annotations)
    assert isinstance(vis_image, np.ndarray), "Visualized image should be a NumPy array"
    assert vis_image.shape == sample_image.shape, "Visualized image should match input image shape"

# --- Tests for SAM2PredictorWrapper ---
def test_sam_predictor_wrapper_loading():
    """Test that SAM2PredictorWrapper initializes correctly."""
    wrapper = SAM2PredictorWrapper(variant='t', max_parallel=1)
    assert wrapper.model is not None, "Model should be loaded"

def test_sam_predictor_wrapper_run(sample_image):
    """Test segmentation with prompts."""
    wrapper = SAM2PredictorWrapper(variant='t', max_parallel=1)
    height, width = sample_image.shape[:2]
    # Define a prompt: a point at the center of the image
    prompts = {
        'point_coords': np.array([[width // 2, height // 2]]),
        'point_labels': np.array([1])  # Positive label
    }
    annotations = wrapper._run(sample_image, prompts)
    assert isinstance(annotations, list), "Annotations should be a list"
    assert len(annotations) > 0, "At least one annotation should be produced"
    assert 'mask' in annotations[0], "Annotation should have a 'mask' key"
    assert 'score' in annotations[0], "Annotation should have a 'score' key"
    assert isinstance(annotations[0]['mask'], np.ndarray), "Mask should be a NumPy array"
    assert annotations[0]['mask'].shape == (height, width), "Mask shape should match image dimensions"

def test_sam_predictor_wrapper_visualize_predictions(sample_image):
    """Test visualization of prompted predictions."""
    wrapper = SAM2PredictorWrapper(variant='t', max_parallel=1)
    height, width = sample_image.shape[:2]
    prompts = {
        'point_coords': np.array([[width // 2, height // 2]]),
        'point_labels': np.array([1])
    }
    annotations = wrapper._run(sample_image, prompts)
    vis_image = wrapper.visualize_predictions(sample_image, annotations)
    assert isinstance(vis_image, np.ndarray), "Visualized image should be a NumPy array"
    assert vis_image.shape == sample_image.shape, "Visualized image should match input image shape"

# --- Tests for Utility Functions ---
def test_get_sam(sample_image):
    """Test the get_sam utility function."""
    sam_processor = get_sam(variant='t', max_parallel=1)
    annotations = sam_processor(sample_image)
    assert isinstance(annotations, list), "Annotations should be a list"
    assert len(annotations) > 0, "At least one annotation should be produced"
    assert 'segmentation' in annotations[0], "Annotation should have a 'segmentation' key"

def test_get_sam_predictor(sample_image):
    """Test the get_sam_predictor utility function."""
    sam_predictor = get_sam_predictor(variant='t', max_parallel=1)
    height, width = sample_image.shape[:2]
    prompts = {
        'point_coords': np.array([[width // 2, height // 2]]),
        'point_labels': np.array([1])
    }
    annotations = sam_predictor(sample_image, prompts)
    assert isinstance(annotations, list), "Annotations should be a list"
    assert len(annotations) > 0, "At least one annotation should be produced"
    assert 'mask' in annotations[0], "Annotation should have a 'mask' key"