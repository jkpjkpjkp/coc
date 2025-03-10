import os
import time
import requests
from PIL import Image
import pytest

# Define test images and objects
TEST_IMAGES = [
    ("data/sample/4girls.jpg", ["person", "face"]),
    ("data/sample/onions.jpg", ["onion", "vegetable"]),
    ("data/sample/img0.png", ["object", "shape"]),
]

# Server URLs
DINO_API_URL = f"http://localhost:{os.environ['dino_port']}/api/predict"
OWL_API_URL = f"http://localhost:{os.environ['owl_port']}/api/predict"
MOD_API_URL = f"http://localhost:{os.environ['grounding_port']}/api/predict"

@pytest.fixture(scope="module", autouse=True)
def start_servers():
    # Import and start servers
    from coc.tool.grounding.mod import launch_all
    import multiprocessing

    # Start servers in background
    server_process = multiprocessing.Process(target=launch_all)
    server_process.start()

    # Wait for servers to start
    time.sleep(10)

    yield

    # Cleanup
    server_process.terminate()

def test_dino_server():
    """Test Grounding DINO server with sample images"""
    for img_path, objects in TEST_IMAGES:
        # Load image
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        # Prepare payload
        payload = {
            "data": [
                {"image": img_bytes.hex(), "mime_type": "image/jpeg"},
                ",".join(objects),
                0.5,  # confidence
                0.2,  # box_threshold
                0.1   # text_threshold
            ]
        }

        # Make request
        response = requests.post(DINO_API_URL, json=payload)
        assert response.status_code == 200

        # Check response
        data = response.json()['data']
        assert len(data) == 3  # image, text, bboxes
        assert "Found" in data[1]  # detection text
        assert isinstance(data[2], list)  # bboxes

def test_owl_server():
    """Test OWLv2 server with sample images"""
    for img_path, objects in TEST_IMAGES:
        # Load image
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        # Prepare payload
        payload = {
            "data": [
                {"image": img_bytes.hex(), "mime_type": "image/jpeg"},
                ",".join(objects),
                0.5,  # confidence
                0.1   # threshold
            ]
        }

        # Make request
        response = requests.post(OWL_API_URL, json=payload)
        assert response.status_code == 200

        # Check response
        data = response.json()['data']
        assert len(data) == 3  # image, text, bboxes
        assert "Found" in data[1]  # detection text
        assert isinstance(data[2], list)  # bboxes

def test_mod_server():
    """Test combined detection server with sample images"""
    for img_path, objects in TEST_IMAGES:
        # Load image
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        # Prepare payload
        payload = {
            "data": [
                {"image": img_bytes.hex(), "mime_type": "image/jpeg"},
                ",".join(objects),
                0.5,  # confidence
                0.2,  # dino_box_threshold
                0.1,  # dino_text_threshold
                0.1   # owl_threshold
            ]
        }

        # Make request
        response = requests.post(MOD_API_URL, json=payload)
        assert response.status_code == 200

        # Check response
        data = response.json()['data']
        assert len(data) == 2  # image, text
        assert "Found" in data[1]  # detection text

def test_server_errors():
    """Test error handling in servers"""
    # Test missing image
    payload = {
        "data": [
            None,
            "person,face",
            0.5,
            0.2,
            0.1
        ]
    }
    response = requests.post(DINO_API_URL, json=payload)
    assert response.status_code == 200
    assert "Please upload an image" in response.json()['data'][1]

    # Test missing objects
    with open("data/sample/4girls.jpg", "rb") as f:
        img_bytes = f.read()
    payload = {
        "data": [
            {"image": img_bytes.hex(), "mime_type": "image/jpeg"},
            "",
            0.5,
            0.2,
            0.1
        ]
    }
    response = requests.post(DINO_API_URL, json=payload)
    assert response.status_code == 200
    assert "Please specify at least one object" in response.json()['data'][1]
