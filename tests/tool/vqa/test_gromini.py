import pytest
from unittest.mock import patch, MagicMock, Mock
from PIL import Image
import numpy as np
import os
import io
import base64
from pathlib import Path
from tenacity import RetryError

from coc.tool.vqa.gromini import Gemini, DEFAULT_MODEL_NAME, DEFAULT_BROKER_POOL

@pytest.fixture
def gemini_tool():
    return Gemini()

@pytest.fixture
def dummy_image():
    # Create a simple 100x100 RGB image
    return Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

# Test initialization of Gemini class
def test_gemini_initialization(gemini_tool):
    assert gemini_tool.name == 'VQA'
    assert "image" in gemini_tool.description and "question" in gemini_tool.description
    assert gemini_tool.mllm is not None
    assert gemini_tool.model_name == DEFAULT_MODEL_NAME
    assert len(gemini_tool.broker_pool) > 0
    assert gemini_tool.current_broker_index == 0
    assert gemini_tool.max_retries > 0
    assert gemini_tool.base_delay > 0
    assert gemini_tool.max_delay > 0
    assert gemini_tool.backoff_factor > 0

# Test custom initialization
def test_custom_initialization():
    custom_broker_pool = [["test_api_key", "test_base_url"]]
    custom_model = "custom-model"
    custom_retries = 10
    custom_delay = 2.0
    custom_max_delay = 60.0
    custom_backoff = 3.0
    
    tool = Gemini(
        max_retries=custom_retries,
        base_delay=custom_delay,
        max_delay=custom_max_delay,
        backoff_factor=custom_backoff,
        broker_pool=custom_broker_pool,
        model_name=custom_model
    )
    
    assert tool.max_retries == custom_retries
    assert tool.base_delay == custom_delay
    assert tool.max_delay == custom_max_delay
    assert tool.backoff_factor == custom_backoff
    assert tool.broker_pool == custom_broker_pool
    assert tool.model_name == custom_model
    assert tool.current_broker_index == 0

# Test image encoding
def test_encode_image(gemini_tool, dummy_image):
    # Test the image encoding functionality
    encoded = gemini_tool._encode_image(dummy_image)

    # Basic checks on the encoded string
    assert isinstance(encoded, str)
    assert len(encoded) > 100  # Base64 encoded image should be reasonably large
    
    # Verify it's a valid base64 string by decoding it
    decoded_image = base64.b64decode(encoded)
    assert decoded_image is not None
    Image.open(io.BytesIO(decoded_image))  # Should not raise an error

# Test broker pool handling
def test_broker_pool_switching():
    # Create a test broker pool with multiple entries
    test_broker_pool = [
        ["api_key_1", "base_url_1"],
        ["api_key_2", "base_url_2"],
        ["api_key_3", "base_url_3"]
    ]
    
    tool = Gemini(broker_pool=test_broker_pool)
    assert tool.current_broker_index == 0
    
    # Test switching to next broker
    result = tool._try_next_broker()
    assert result is True
    assert tool.current_broker_index == 1
    
    # Switch again
    result = tool._try_next_broker()
    assert result is True
    assert tool.current_broker_index == 2
    
    # Switch again - should wrap around to 0
    result = tool._try_next_broker()
    assert result is True
    assert tool.current_broker_index == 0

# Test single broker pool behavior
def test_single_broker_pool():
    test_broker_pool = [["api_key_1", "base_url_1"]]
    tool = Gemini(broker_pool=test_broker_pool)
    
    # Switching should return False when only one broker is available
    result = tool._try_next_broker()
    assert result is False
    assert tool.current_broker_index == 0

# Test handling of run with retry using mocks
@patch('coc.tool.vqa.gromini.HumanMessage')
@patch('coc.tool.vqa.gromini.ChatOpenAI')
def test_run_with_retry(mock_chat, mock_human_msg, gemini_tool, dummy_image):
    # Configure mock
    mock_response = MagicMock()
    mock_response.content = "Test response"
    mock_chat_instance = MagicMock()
    mock_chat_instance.invoke.return_value = mock_response
    mock_chat.return_value = mock_chat_instance
    
    # Configure human message mock
    mock_msg_instance = MagicMock()
    mock_human_msg.return_value = mock_msg_instance
    
    # Call the method
    response = gemini_tool._run_with_retry(dummy_image, "Test question")
    
    # Assert the response
    assert response == "Test response"
    
    # Verify HumanMessage was created correctly
    mock_human_msg.assert_called_once()
    
    # Verify invoke was called
    mock_chat_instance.invoke.assert_called_once()

# Test handling of multiple images using mocks
@patch('coc.tool.vqa.gromini.HumanMessage')
@patch('coc.tool.vqa.gromini.ChatOpenAI')
def test_run_multiimage(mock_chat, mock_human_msg, gemini_tool):
    # Create test images
    test_images = [
        Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)),
        Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 255)
    ]
    
    # Configure mock
    mock_response = MagicMock()
    mock_response.content = "Test multiple images response"
    mock_chat_instance = MagicMock()
    mock_chat_instance.invoke.return_value = mock_response
    mock_chat.return_value = mock_chat_instance
    
    # Configure human message mock
    mock_msg_instance = MagicMock()
    mock_human_msg.return_value = mock_msg_instance
    
    # Call the method
    with patch.object(gemini_tool, '_run_multiimage_with_retry') as mock_run_with_retry:
        mock_run_with_retry.return_value = "Test multiple images response"
        response = gemini_tool._run_multiimage(test_images, "Test multiple images question")
    
    # Assert the response
    assert response == "Test multiple images response"

# Test handling non-image input in run_freestyle
@patch('coc.tool.vqa.gromini.HumanMessage')
@patch('coc.tool.vqa.gromini.ChatOpenAI')
def test_run_freestyle_text_only(mock_chat, mock_human_msg, gemini_tool):
    # Configure mock
    mock_response = MagicMock()
    mock_response.content = "Text only response"
    mock_chat_instance = MagicMock()
    mock_chat_instance.invoke.return_value = mock_response
    mock_chat.return_value = mock_chat_instance
    
    # Configure human message mock
    mock_msg_instance = MagicMock()
    mock_human_msg.return_value = mock_msg_instance
    
    # Mock invoke method
    with patch.object(gemini_tool, 'invoke') as mock_invoke:
        mock_invoke.return_value = "Text only response"
        response = gemini_tool.run_freestyle(["Text only prompt"])
    
    # Assert the response
    assert response == "Text only response"

# Test error handling with broken broker
@patch('coc.tool.vqa.gromini.HumanMessage')
def test_broker_failure_handling(mock_human_msg, dummy_image):
    # Create a broker pool with one broker that will fail
    test_broker_pool = [
        ["api_key_1", "bad_url_1"],
        ["api_key_2", "good_url_2"]
    ]
    
    # Configure human message mock
    mock_msg_instance = MagicMock()
    mock_human_msg.return_value = mock_msg_instance
    
    tool = Gemini(broker_pool=test_broker_pool)
    
    # Make the first broker fail, second broker succeed
    with patch.object(tool, '_run_with_retry') as mock_run_with_retry:
        # First call raises exception, second call succeeds
        mock_run_with_retry.side_effect = [Exception("API Error"), "Success response"]
        
        # This should succeed using the second broker
        response = tool._run(dummy_image, "Test question")
        
        # Assert we got the success response
        assert response == "Success response"
        assert tool.current_broker_index == 1  # Should have switched to second broker 