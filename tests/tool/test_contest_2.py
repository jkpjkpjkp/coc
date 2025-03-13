import pytest
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Union
from coc.tool.context import (grounding, grounding_dino, owl, glm, qwen, gemini,
                        segment_anything, depth_anything, google_search, Bbox)

# Note: Replace 'your_module' with the actual module name where the interface is defined.

def test_grounding():
    """Test the grounding function."""
    # Create a dummy image and input
    image = Image.new('RGB', (100, 100))
    objects_of_interest = ["cat", "dog"]

    # Call the function
    result = grounding(image, objects_of_interest)

    # Assert return type and structure
    assert isinstance(result, tuple), "Result should be a tuple"
    assert len(result) == 3, "Result should have 3 elements"
    assert isinstance(result[0], Image.Image), "First element should be a PIL Image"
    assert isinstance(result[1], str), "Second element should be a string"
    assert isinstance(result[2], list), "Third element should be a list"
    for bbox in result[2]:
        assert isinstance(bbox, dict), "Each bbox should be a dictionary"
        assert 'box' in bbox, "Bbox should have 'box' key"
        assert 'score' in bbox, "Bbox should have 'score' key"
        assert 'label' in bbox, "Bbox should have 'label' key"
        assert isinstance(bbox['box'], list), "'box' should be a list"
        assert len(bbox['box']) == 4, "'box' should have 4 elements"
        assert all(isinstance(x, float) for x in bbox['box']), "'box' elements should be floats"
        assert isinstance(bbox['score'], float), "'score' should be a float"
        assert isinstance(bbox['label'], str), "'label' should be a string"

def test_grounding_dino():
    """Test the grounding_dino function."""
    image = Image.new('RGB', (100, 100))
    objects_of_interest = ["cat", "dog"]
    result = grounding_dino(image, objects_of_interest)

    # Same assertions as grounding since return type is identical
    assert isinstance(result, tuple) and len(result) == 3
    assert isinstance(result[0], Image.Image)
    assert isinstance(result[1], str)
    assert isinstance(result[2], list)
    for bbox in result[2]:
        assert isinstance(bbox, dict)
        assert all(key in bbox for key in ['box', 'score', 'label'])
        assert isinstance(bbox['box'], list) and len(bbox['box']) == 4
        assert all(isinstance(x, float) for x in bbox['box'])
        assert isinstance(bbox['score'], float)
        assert isinstance(bbox['label'], str)

def test_owl():
    """Test the owl function."""
    image = Image.new('RGB', (100, 100))
    objects_of_interest = ["cat", "dog"]
    result = owl(image, objects_of_interest)

    # Same assertions as grounding
    assert isinstance(result, tuple) and len(result) == 3
    assert isinstance(result[0], Image.Image)
    assert isinstance(result[1], str)
    assert isinstance(result[2], list)
    for bbox in result[2]:
        assert isinstance(bbox, dict)
        assert all(key in bbox for key in ['box', 'score', 'label'])
        assert isinstance(bbox['box'], list) and len(bbox['box']) == 4
        assert all(isinstance(x, float) for x in bbox['box'])
        assert isinstance(bbox['score'], float)
        assert isinstance(bbox['label'], str)

def test_glm():
    """Test the glm VQA function."""
    image = Image.new('RGB', (100, 100))
    question = "What is in the image?"
    result = glm(image, question)

    assert isinstance(result, str), "Result should be a string"

def test_qwen():
    """Test the qwen VQA function."""
    image = Image.new('RGB', (100, 100))
    question = "What is in the image?"
    result = qwen(image, question)

    assert isinstance(result, str), "Result should be a string"

def test_gemini():
    """Test the gemini VQA function with various input combinations."""
    image = Image.new('RGB', (100, 100))
    question = "What is in the image?"

    # Test with one image and one question
    result = gemini(image, question)
    assert isinstance(result, str), "Result should be a string with image and question"

    # Test with multiple images
    result = gemini(image, image)
    assert isinstance(result, str), "Result should be a string with multiple images"

    # Test with text only
    result = gemini("Hello")
    assert isinstance(result, str), "Result should be a string with text only"

    # Test with mixed inputs
    result = gemini(image, "Question?", image)
    assert isinstance(result, str), "Result should be a string with mixed inputs"

def test_segment_anything():
    """Test the segment_anything function."""
    image = Image.new('RGB', (100, 100))
    result = segment_anything(image)

    assert isinstance(result, list), "Result should be a list"
    for item in result:
        assert isinstance(item, dict), "Each item should be a dictionary"
        # Note: Specific keys are not specified in the interface, so we only check type

def test_depth_anything():
    """Test the depth_anything function."""
    image = Image.new('RGB', (100, 100))
    result = depth_anything(image)

    assert isinstance(result, np.ndarray), "Result should be a NumPy array"
    assert result.shape == (100, 100), "Result should have shape HxW matching input image"

def test_google_search():
    """Test the google_search function."""
    query = "test query"
    result = google_search(query)

    assert isinstance(result, list), "Result should be a list"
    for item in result:
        assert isinstance(item, str), "Each item should be a string"

# Optional: Test error handling for invalid inputs
def test_grounding_invalid_image():
    """Test grounding with invalid image input."""
    with pytest.raises(TypeError):
        grounding("not an image", ["cat"])

def test_glm_invalid_question():
    """Test glm with invalid question type."""
    image = Image.new('RGB', (100, 100))
    with pytest.raises(TypeError):
        glm(image, 123)