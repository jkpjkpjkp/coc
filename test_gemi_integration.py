#!/usr/bin/env python3
"""
Integration test for GeminiAgent

This script verifies that the GeminiAgent works as expected with all components
properly integrated. It mocks external dependencies to focus on internal logic.
"""

import unittest
import os
import sys
import logging
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np

# Set up logging
logging.basicConfig(level=logging.WARNING)

# Add the parent directory to the path to import coc modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from coc.tree.gemi import GeminiAgent
from coc.exec.mod import Exec

class GeminiIntegrationTest(unittest.TestCase):
    """Integration tests for GeminiAgent"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        
        # Patch external dependencies
        self.patcher1 = patch('coc.tool.vqa.gemini.Gemini')
        self.patcher2 = patch('coc.tree.gemi.Exec')
        
        # Start the patchers
        self.mock_gemini_class = self.patcher1.start()
        self.mock_exec_class = self.patcher2.start()
        
        # Set up the mock Gemini instance
        self.mock_gemini = MagicMock()
        self.mock_gemini.run_freestyle.return_value = "Mocked Gemini response"
        self.mock_gemini_class.return_value = self.mock_gemini
        
        # Set up the mock Exec instance
        self.mock_exec = MagicMock()
        self.mock_exec._run.return_value = ("Executed code output", "", [])
        self.mock_exec_class.return_value = self.mock_exec
        
        # Create GeminiAgent with mocked dependencies
        self.agent = GeminiAgent(
            use_depth=True, 
            use_segmentation=True,
            use_novel_view=False,
            use_point_cloud=False,
            verbose=True
        )
        
        # Mock modules_available to ensure our mocked capabilities are "available"
        self.agent.modules_available = {
            "depth": True,
            "segmentation": True,
            "novel_view": False,
            "point_cloud": False
        }
    
    def tearDown(self):
        """Clean up patchers"""
        self.patcher1.stop()
        self.patcher2.stop()
    
    def test_generate_orchestrated_with_counting_prompt(self):
        """Test the generate_orchestrated method with a counting prompt"""
        # Configure mock Gemini responses for code execution path
        self.mock_gemini.run_freestyle.side_effect = [
            "```python\nprint('Counting bottles...')\nprint('Found 23 bottles')\n```",  # Code generation
            "There are 23 bottles in the image."  # Interpretation
        ]
        
        # Call the method with a counting task
        result = self.agent.generate_orchestrated("How many bottles are in this image?", [self.test_image])
        
        # Verify the call flow
        self.assertEqual(self.mock_gemini.run_freestyle.call_count, 2)
        self.assertEqual(self.mock_exec._run.call_count, 1)
        
        # Verify the images were added to the environment
        self.mock_exec.set_var.assert_any_call("image_0", self.test_image)
        
        # Check the result includes the interpretation
        self.assertEqual(result, "Executed code output\n\nThere are 23 bottles in the image.")
    
    def test_generate_with_standard_prompt(self):
        """Test the generate method with a standard (non-counting) prompt"""
        # Call the method
        result = self.agent.generate("Describe this image", [self.test_image])
        
        # Verify Gemini was called
        self.mock_gemini.run_freestyle.assert_called_once()
        
        # Check the result
        self.assertEqual(result, "Mocked Gemini response")

if __name__ == "__main__":
    unittest.main() 