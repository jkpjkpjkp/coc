import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np
from coc.tool.vqa.gemini import Gemini
from coc.tool.task import Bbox

@pytest.fixture
def gemini_tool():
    return Gemini()

@pytest.fixture
def dummy_image():
    # Create a simple 100x100 RGB image
    return Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

def test_gemini_image_encoding(gemini_tool, dummy_image):
    # Test the image encoding functionality
    encoded = gemini_tool._encode_image(dummy_image)

    # Basic checks on the encoded string
    assert isinstance(encoded, str)
    assert len(encoded) > 100  # Base64 encoded image should be reasonably large