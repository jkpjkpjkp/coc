from PIL import Image as Img
from typing import List, Optional, Dict
from coc.exec.context.mod import grounding, grounding_dino, owl, glm, qwen, gemini  # Assuming the interface is in interface.py

def create_dummy_image(width: int = 100, height: int = 100) -> Img:
    """Create a dummy white image for testing."""
    return Img.new('RGB', (width, height), color='white')

def check_bbox_list(result: List[Dict]) -> None:
    """Helper function to validate that the result is a list of Bbox dictionaries."""
    assert isinstance(result, list), "Result should be a list"
    for item in result:
        assert isinstance(item, dict), "Each item should be a dictionary"
        assert 'box' in item, "Dictionary should have 'box' key"
        assert isinstance(item['box'], list), "'box' should be a list"
        assert len(item['box']) == 4, "'box' should have 4 elements"
        assert all(isinstance(x, float) for x in item['box']), "'box' elements should be floats"
        assert 'score' in item, "Dictionary should have 'score' key"
        assert isinstance(item['score'], float), "'score' should be a float"
        assert 'label' in item, "Dictionary should have 'label' key"
        assert isinstance(item['label'], str), "'label' should be a string"

# Grounding Tools Tests

def test_grounding():
    """Test the grounding function with and without objects_of_interest."""
    image = create_dummy_image()
    objects_of_interest = ['cat', 'dog']
    result = grounding(image, objects_of_interest)
    assert result == []
    
def test_grounding_dino():
    """Test the grounding_dino function with and without objects_of_interest."""
    image = create_dummy_image()
    objects_of_interest = ['cat', 'dog']
    result = grounding_dino(image, objects_of_interest)
    check_bbox_list(result[2])

def test_owl():
    """Test the owl function with and without objects_of_interest."""
    image = create_dummy_image()
    objects_of_interest = ['cat', 'dog']
    result = owl(image, objects_of_interest)
    check_bbox_list(result[2])

# VQA Tools Tests

def test_glm():
    """Test the glm function with a sample question."""
    image = create_dummy_image()
    question = "What is in the image?"
    result = glm(image, question)
    assert isinstance(result, str), "Result should be a string"

def test_qwen():
    """Test the qwen function with a sample question."""
    image = create_dummy_image()
    question = "What is in the image?"
    result = qwen(image, question)
    assert isinstance(result, str), "Result should be a string"

def test_gemini():
    """Test the gemini function with a sample question."""
    image = create_dummy_image()
    question = "What is in the image?"
    result = gemini(image, question)
    assert isinstance(result, str), "Result should be a string"

# To execute these tests, save this file (e.g., as test_interface.py) and run:
# pytest test_interface.py
# Alternatively, run each function manually to check assertions.