import pytest
from PIL import Image
import numpy as np
import torch
import cv2
import sys
sys.modules['cv2'] = cv2  # Handle module import issues if any

# Import the module containing DepthFactory and get_depth
from coc.tool.depth import DepthFactory, get_depth, _depth_factory, _depth_lock
ENCODER = 'vitl'
# Fixtures for test images
@pytest.fixture
def sample_image_path(tmp_path):
    img_path = tmp_path / "test_image.jpg"
    img = Image.new('RGB', (100, 100), color='red')
    img.save(img_path)
    return str(img_path)

@pytest.fixture
def sample_pil_image():
    return Image.new('RGB', (100, 100), color='blue')

# Test DepthFactory initialization
def test_initialization_valid_encoders():
    valid_encoders = ['vitl']
    for encoder in valid_encoders:
        try:
            DepthFactory(encoder=encoder)
        except Exception as e:
            pytest.fail(f"Initialization failed for encoder {encoder}: {e}")

def test_initialization_invalid_encoder():
    with pytest.raises(ValueError, match="Encoder must be one of"):
        DepthFactory(encoder='invalid')

def test_device_selection():
    # Auto-detection
    factory = DepthFactory()
    assert factory.device in ['cuda', 'cpu']
    # Explicit device (if CUDA available)
    if torch.cuda.is_available():
        factory = DepthFactory(device='cuda')
        assert factory.device == 'cuda'
    else:
        factory = DepthFactory(device='cuda')
        assert factory.device == 'cpu'

# Test _run method
def test_run_with_image_path(sample_image_path):
    factory = DepthFactory(encoder=ENCODER)
    depth_map = factory._run(sample_image_path)
    assert isinstance(depth_map, np.ndarray)
    assert depth_map.ndim == 2
    assert depth_map.shape == (100, 100)  # Assuming model maintains size

def test_run_with_pil_image(sample_pil_image):
    factory = DepthFactory(encoder=ENCODER)
    depth_map = factory._run(sample_pil_image)
    assert isinstance(depth_map, np.ndarray)
    assert depth_map.ndim == 2

def test_run_invalid_input():
    factory = DepthFactory()
    with pytest.raises(ValueError):
        factory._run(123)  # Invalid input type

# Test get_depth function
def test_get_depth_returns_callable():
    estimator = get_depth()
    assert callable(estimator)

def test_process_depth_with_image_path(sample_image_path):
    estimator = get_depth(encoder=ENCODER)
    depth_map = estimator(sample_image_path)
    assert isinstance(depth_map, np.ndarray)

def test_process_depth_with_pil_image(sample_pil_image):
    estimator = get_depth(encoder=ENCODER)
    depth_map = estimator(sample_pil_image)
    assert isinstance(depth_map, np.ndarray)

def test_error_propagation():
    estimator = get_depth()
    with pytest.raises(RuntimeError):
        estimator('non_existent_image.jpg')