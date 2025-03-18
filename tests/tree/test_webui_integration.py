import unittest
import os
import sys
import requests
from PIL import Image
import io
import base64
import tempfile
from unittest import skipIf

from coc.tree.qua import WebUIWrapper
from coc.tree.webui_config import WEBUI_API_BASE_URL, DEFAULT_MODEL

# Check if the WebUI API is available
WEBUI_AVAILABLE = False
try:
    response = requests.get(f"{WEBUI_API_BASE_URL}/api/models", timeout=2)
    WEBUI_AVAILABLE = response.status_code == 200
except:
    pass

# Get API key from environment if available
API_KEY = os.environ.get("WEBUI_API_KEY", "")


@skipIf(not WEBUI_AVAILABLE, "WebUI API is not available")
class TestWebUIIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        # Create a simple test image
        cls.test_image = Image.new('RGB', (100, 100), color='red')
        
        # Create a temporary file for the test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            cls.image_path = temp_file.name
            cls.test_image.save(cls.image_path)
            
        # Create WebUIWrapper instance
        cls.webui = WebUIWrapper(base_url=WEBUI_API_BASE_URL)
        
        # Add API key to headers if available
        if API_KEY:
            # Note: This is simplified - in a real implementation, 
            # you'd need to properly integrate the API key handling
            os.environ["WEBUI_API_KEY"] = API_KEY
    
    @classmethod
    def tearDownClass(cls):
        """Tear down test fixtures once for all tests."""
        # Remove temporary image file
        if os.path.exists(cls.image_path):
            os.remove(cls.image_path)
    
    def test_encode_image(self):
        """Test image encoding functionality."""
        # Test with PIL Image
        encoded = self.webui._encode_image(self.test_image)
        self.assertTrue(encoded.startswith("data:image/png;base64,"))
        
        # Test with image path
        encoded_path = self.webui._encode_image(self.image_path)
        self.assertEqual(encoded_path, self.image_path)
    
    @skipIf(not API_KEY, "API key not available")
    def test_generate_simple_text(self):
        """Test generation with simple text prompt."""
        response = self.webui.generate("What is 2+2?")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    @skipIf(not API_KEY, "API key not available")
    def test_generate_with_image(self):
        """Test generation with image input."""
        response = self.webui.generate(
            "Describe this image in one sentence.", 
            images=[self.test_image]
        )
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    @skipIf(not API_KEY, "API key not available")
    def test_run_freestyle(self):
        """Test run_freestyle interface."""
        response = self.webui.run_freestyle(["What is the capital of France?"])
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    @skipIf(not API_KEY, "API key not available")
    def test_generate_branches(self):
        """Test parallel generation of multiple responses."""
        prompts = [
            "What is the capital of France?",
            "What is the capital of Italy?"
        ]
        responses = self.webui.generate_branches(prompts)
        
        # Verify results
        self.assertEqual(len(responses), 2)
        self.assertIsInstance(responses[0], str)
        self.assertIsInstance(responses[1], str)
        self.assertGreater(len(responses[0]), 0)
        self.assertGreater(len(responses[1]), 0)


if __name__ == '__main__':
    unittest.main() 