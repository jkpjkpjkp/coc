# test_api.py
from gradio_client import Client, handle_file
from PIL import Image
import unittest
from coc.config import dino_port

class TestAPI(unittest.TestCase):
    def setUp(self):
        # Initialize the Gradio client for each test
        # Assumes the server is running at http://localhost:{dino_port}
        self.client = Client(f"http://localhost:{dino_port}")

    def test_predict_valid_input(self):
        # Test with a valid image and input
        image_path = "data/sample/4girls.jpg"  # Replace with a real test image path
        result = self.client.predict(
            image=handle_file(image_path),
            object_list_text="cat, dog, person",
            confidence=0.5,
            box_threshold=0.2,
            text_threshold=0.1,
            api_name="/predict"
        )
        print(result)
        self.assertIsInstance(result[0], str)  # Output image path
        self.assertIsInstance(result[1], str)  # Detection details text
        self.assertIsInstance(result[2], list)  # Detections list
        self.assertTrue("Found" in result[1] or "No objects detected" in result[1])

    def test_predict_no_objects(self):
        # Test with no objects specified
        image_path = "data/sample/onions.jpg"  # Replace with a real test image path
        result = self.client.predict(
            image=handle_file(image_path),
            object_list_text="",
            confidence=0.2,
            box_threshold=0.2,
            text_threshold=0.1,
            api_name="/predict"
        )
        print(result)
        self.assertIsInstance(result[0], str)  # Image path returned unchanged
        self.assertIn("Please specify at least one object", result[1])  # Error message
        self.assertEqual(result[2], [])  # Empty detections list

if __name__ == "__main__":
    unittest.main()