#!/usr/bin/env python3
"""
Test script for image loading functionality in run_gemi.py

This script tests the image loading and path handling in the process_image function
to identify the cause of the "Image file not found" error.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import logging
import tempfile

# Add parent directory to path to import run_gemi
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to test
from run_gemi import process_image, GeminiAgent

class TestImageLoading(unittest.TestCase):
    """Test cases for image loading functionality in run_gemi.py"""
    
    def setUp(self):
        """Set up test environment with temporary test images"""
        # Create a temporary directory for test images
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a simple test image
        from PIL import Image
        import numpy as np
        
        # Create a small red square image
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img_array[:, :, 0] = 255  # Red channel
        self.test_image = Image.fromarray(img_array)
        
        # Save the test image to the temp directory
        self.test_image_path = os.path.join(self.temp_dir.name, "test_image.jpg")
        self.test_image.save(self.test_image_path)
        
        # Create logging directory if it doesn't exist
        os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'data', 'log'), exist_ok=True)
    
    def tearDown(self):
        """Clean up temporary files"""
        self.temp_dir.cleanup()
    
    @patch('run_gemi.GeminiAgent')
    def test_process_image_with_absolute_path(self, mock_gemini_agent):
        """Test that process_image works with absolute paths"""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.generate.return_value = "Test response"
        mock_gemini_agent.return_value = mock_instance
        
        # Run the function with absolute path
        result = process_image(self.test_image_path, "Test prompt")
        
        # Verify the function worked
        self.assertTrue(result)
        mock_instance.generate.assert_called_once()
    
    @patch('run_gemi.GeminiAgent')
    def test_process_image_with_relative_path(self, mock_gemini_agent):
        """Test that process_image works with relative paths"""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.generate.return_value = "Test response"
        mock_gemini_agent.return_value = mock_instance
        
        # Get the current directory
        current_dir = os.getcwd()
        
        # Change to parent directory of test image
        os.chdir(os.path.dirname(self.test_image_path))
        
        # Use relative path
        relative_path = os.path.basename(self.test_image_path)
        
        try:
            # Run the function with relative path
            result = process_image(relative_path, "Test prompt")
            
            # Verify the function worked
            self.assertTrue(result)
            mock_instance.generate.assert_called_once()
        finally:
            # Restore current directory
            os.chdir(current_dir)
    
    def test_file_not_found_error(self):
        """Test the error handling when image file is not found"""
        # Test with non-existent file
        non_existent_path = "image_0.jpg"
        
        # Run the function with non-existent path
        result = process_image(non_existent_path, "Test prompt")
        
        # Should return False but not raise an exception
        self.assertFalse(result)
    
    @patch('run_gemi.GeminiAgent')
    def test_zerobench_image_path(self, mock_gemini_agent):
        """Test with a path similar to the one in the error message"""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.generate.return_value = "Test response"
        mock_gemini_agent.return_value = mock_instance
        
        # Create directory structure similar to the one in the error
        os.makedirs(os.path.join(self.temp_dir.name, "zerobench"), exist_ok=True)
        
        # Create test image with same name as in error
        test_path = os.path.join(self.temp_dir.name, "zerobench", "example_0_image_0.png")
        self.test_image.save(test_path)
        
        # Run the function with the test path
        result = process_image(test_path, "Test prompt")
        
        # Verify the function worked
        self.assertTrue(result)
        mock_instance.generate.assert_called_once()

if __name__ == "__main__":
    unittest.main() 