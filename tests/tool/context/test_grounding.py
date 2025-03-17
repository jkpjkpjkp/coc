import pytest
from PIL import Image
from coc.exec.context.mod import grounding, grounding_dino, owl

def check_bbox_list(bbox_list, objects_of_interest):
    """
    Helper function to verify that the bounding box list has the correct structure and content.

    Args:
        bbox_list: List of dictionaries containing bounding box data.
        objects_of_interest: List of strings specifying the objects to detect.
    """
    assert isinstance(bbox_list, list), "Output should be a list"
    for bbox in bbox_list:
        assert isinstance(bbox, dict), "Each bounding box should be a dictionary"
        assert set(bbox.keys()) == {'box', 'score', 'label'}, "Bbox dictionary should have exactly 'box', 'score', 'label' keys"
        assert isinstance(bbox['box'], list), "'box' should be a list"
        assert len(bbox['box']) == 4, "'box' should contain 4 coordinates"
        assert all(isinstance(x, (float, int)) for x in bbox['box']), "'box' coordinates should be numbers"
        assert isinstance(bbox['score'], (float, int)), "'score' should be a number"
        assert 0 <= bbox['score'] <= 1, "'score' should be between 0 and 1"
        assert isinstance(bbox['label'], str), "'label' should be a string"
        assert bbox['label'] in objects_of_interest, "'label' should be one of the objects of interest"

# Parametrized tests for grounding_dino
@pytest.mark.parametrize("image_path, objects_of_interest", [
    ("data/sample/onions.jpg", ["onion"]),
    ("data/sample/4girls.jpg", ["girl"]),
])
def test_grounding_dino(image_path, objects_of_interest):
    """
    Test the grounding_dino function with different images and objects of interest.
    Verifies that it returns a non-empty list of correctly structured bounding boxes.
    """
    image = Image.open(image_path)
    bbox_list = grounding_dino(image, objects_of_interest)
    check_bbox_list(bbox_list[2], objects_of_interest)
    assert len(bbox_list) > 0, "Should detect at least one object"

# Parametrized tests for owl
@pytest.mark.parametrize("image_path, objects_of_interest", [
    ("data/sample/onions.jpg", ["onion"]),
    ("data/sample/4girls.jpg", ["girl"]),
])
def test_owl(image_path, objects_of_interest):
    """
    Test the owl function with different images and objects of interest.
    Verifies that it returns a non-empty list of correctly structured bounding boxes.
    """
    image = Image.open(image_path)
    bbox_list = owl(image, objects_of_interest)
    check_bbox_list(bbox_list[2], objects_of_interest)
    assert len(bbox_list) > 0, "Should detect at least one object"

# Parametrized tests for grounding
@pytest.mark.parametrize("image_path, objects_of_interest", [
    ("data/sample/onions.jpg", ["onion"]),
    ("data/sample/4girls.jpg", ["girl"]),
])
def test_grounding(image_path, objects_of_interest):
    """
    Test the grounding function with different images and objects of interest.
    Verifies that it returns a non-empty list of correctly structured bounding boxes.
    """
    image = Image.open(image_path)
    bbox_list = grounding(image, objects_of_interest)
    assert isinstance(bbox_list, list)
    assert isinstance(bbox_list[0], tuple)

# Tests for empty objects_of_interest
def test_grounding_dino_empty_objects():
    """
    Test grounding_dino with an empty objects_of_interest list.
    Expects an empty list of bounding boxes.
    """
    image = Image.open("data/sample/onions.jpg")
    objects_of_interest = []
    bbox_list = grounding_dino(image, objects_of_interest)
    assert bbox_list == [], "Should return an empty list when no objects are specified"

def test_owl_empty_objects():
    """
    Test owl with an empty objects_of_interest list.
    Expects an empty list of bounding boxes.
    """
    image = Image.open("data/sample/onions.jpg")
    objects_of_interest = []
    bbox_list = owl(image, objects_of_interest)
    assert bbox_list == [], "Should return an empty list when no objects are specified"

def test_grounding_empty_objects():
    """
    Test grounding with an empty objects_of_interest list.
    Expects an empty list of bounding boxes.
    """
    image = Image.open("data/sample/onions.jpg")
    objects_of_interest = []
    bbox_list = grounding(image, objects_of_interest)
    assert bbox_list == [], "Should return an empty list when no objects are specified"