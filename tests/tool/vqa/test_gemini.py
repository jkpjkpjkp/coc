import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np
from coc.tool.vqa.gemini import Gemini
from coc.tool.context import Bbox

@pytest.fixture
def gemini_tool():
    return Gemini()

@pytest.fixture
def dummy_image():
    # Create a simple 100x100 RGB image
    return Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

def test_gemini_single_image_vqa(gemini_tool, dummy_image):
    # Mock the LLM response
    mock_response = MagicMock()
    mock_response.content = "Test response"
    
    with patch.object(gemini_tool.mllm, 'invoke', return_value=mock_response) as mock_invoke:
        question = "What is in this image?"
        response = gemini_tool._run(dummy_image, question)
        
        # Verify the response
        assert response == "Test response"
        mock_invoke.assert_called_once()

def test_gemini_multi_image_vqa(gemini_tool, dummy_image):
    # Mock the LLM response
    mock_response = MagicMock()
    mock_response.content = "Multi-image response"
    
    with patch.object(gemini_tool.mllm, 'invoke', return_value=mock_response) as mock_invoke:
        question = "Describe these images"
        images = [dummy_image, dummy_image, dummy_image]
        response = gemini_tool._run_multiimage(images, question)
        
        # Verify the response
        assert response == "Multi-image response"
        mock_invoke.assert_called_once()

def test_gemini_image_encoding(gemini_tool, dummy_image):
    # Test the image encoding functionality
    encoded = gemini_tool._encode_image(dummy_image)
    
    # Basic checks on the encoded string
    assert isinstance(encoded, str)
    assert len(encoded) > 100  # Base64 encoded image should be reasonably large

def test_gemini_run_freestyle(gemini_tool, dummy_image):
    # Mock the LLM response
    mock_response = MagicMock()
    mock_response.content = "Freestyle response"
    
    with patch.object(gemini_tool.mllm, 'invoke', return_value=mock_response) as mock_invoke:
        # Test with mixed text and image paths
        prompt = ["text1", "text2", "data/sample/onions.jpg"]
        response = gemini_tool.run_freestyle(prompt)
        
        assert response == "Freestyle response"
        mock_invoke.assert_called_once()

def test_gemini_error_handling(gemini_tool):
    # Test error handling in _run method
    with patch.object(gemini_tool.mllm, 'invoke', side_effect=Exception("Test error")):
        with pytest.raises(Exception):
            gemini_tool._run("invalid_image", "test question")

def test_gemini_multi_image_error_handling(gemini_tool):
    # Test error handling in _run_multiimage method
    with patch.object(gemini_tool.mllm, 'invoke', side_effect=Exception("Test error")):
        with pytest.raises(Exception):
            gemini_tool._run_multiimage(["invalid_image1", "invalid_image2"], "test question")
