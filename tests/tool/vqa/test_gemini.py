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

import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import patch, Mock
import base64
import io
# Fixture to create a Gemini instance with mocked environment variables
@pytest.fixture
def gemini_tool(monkeypatch):
    # Mock environment variables to avoid requiring actual API keys
    monkeypatch.setenv("GEMINI_API_KEY", "mocked_api_key")
    monkeypatch.setenv("GEMINI_BASE_URL", "http://mocked.base.url")
    return Gemini()

# Test initialization of Gemini class
def test_gemini_initialization(gemini_tool):
    assert gemini_tool.name == 'VQA'
    assert gemini_tool.description.startswith('performs VQA on an image.')
    assert gemini_tool.mllm is not None
    assert gemini_tool.mllm.model_name == 'gemini-2.0-pro-exp-02-05'
    assert gemini_tool.LOGFILE == 'data/log/big_gemini.log'

# Test image encoding
def test_encode_image(gemini_tool):
    sample_image_path = Path("data/sample/onions.jpg")
    assert sample_image_path.exists(), f"Sample image {sample_image_path} not found."
    image = Image.open(sample_image_path)
    encoded_image = gemini_tool._encode_image(image)
    assert isinstance(encoded_image, str)
    assert len(encoded_image) > 0
    # Verify it's a valid base64 string by decoding it
    decoded_image = base64.b64decode(encoded_image)
    assert decoded_image is not None
    Image.open(io.BytesIO(decoded_image))  # Should not raise an error