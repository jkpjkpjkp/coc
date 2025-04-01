#!/usr/bin/env python3
"""
Test script to diagnose the OpenCV color conversion error

This test reproduces the error: 
cv2.error: OpenCV(4.11.0) ... error: (-15:Bad number of channels)
"""

import unittest
import numpy as np
import cv2
from PIL import Image
import os
import sys
import tempfile
from coc.util.logging import get_logger

# Get logger instance
logger = get_logger(__name__)

class TestColorConversion(unittest.TestCase):
    """Test cases for OpenCV color conversion errors"""
    
    def setUp(self):
        """Set up test environment with sample images"""
        # Create test images with different channel counts
        self.rgb_array = np.zeros((100, 100, 3), dtype=np.uint8)
        self.rgb_array[:, :, 0] = 255  # Red channel
        self.rgb_image = Image.fromarray(self.rgb_array)
        
        # Create grayscale image (1 channel)
        self.gray_array = np.zeros((100, 100), dtype=np.uint8)
        self.gray_array[:, :] = 128  # 50% gray
        self.gray_image = Image.fromarray(self.gray_array, mode='L')
        
        # Create temp directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.rgb_path = os.path.join(self.temp_dir.name, "rgb_test.jpg")
        self.gray_path = os.path.join(self.temp_dir.name, "gray_test.jpg")
        
        # Save test images
        self.rgb_image.save(self.rgb_path)
        self.gray_image.save(self.gray_path)
    
    def tearDown(self):
        """Clean up temporary files"""
        self.temp_dir.cleanup()
    
    def test_reproduce_error(self):
        """Reproduce the exact error from the logs"""
        # Load grayscale image
        gray_cv = cv2.imread(self.gray_path, cv2.IMREAD_GRAYSCALE)
        
        # Verify it's 1 channel
        self.assertEqual(len(gray_cv.shape), 2)
        
        # This would fail with the same error in the logs
        with self.assertRaises(cv2.error) as context:
            # Attempt to convert grayscale to grayscale - this causes the error
            cv2.cvtColor(gray_cv, cv2.COLOR_BGR2GRAY)
        
        # Verify error message contains the expected text
        self.assertIn("Invalid number of channels", str(context.exception))
    
    def test_safe_conversion(self):
        """Test a safe conversion approach that avoids the error"""
        # Load grayscale image
        gray_cv = cv2.imread(self.gray_path, cv2.IMREAD_GRAYSCALE)
        
        # Function to safely convert to grayscale
        def safe_convert_to_gray(image):
            """Safely convert an image to grayscale, handling already-grayscale images"""
            # Check if image is already grayscale (2D array)
            if len(image.shape) == 2:
                return image
            # Convert from BGR/RGB to grayscale
            elif len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
        
        # This should work without error
        result = safe_convert_to_gray(gray_cv)
        self.assertEqual(len(result.shape), 2)
        
        # Test with RGB image too
        rgb_cv = cv2.imread(self.rgb_path)
        result_rgb = safe_convert_to_gray(rgb_cv)
        self.assertEqual(len(result_rgb.shape), 2)
    
    def test_count_objects_function(self):
        """Test a simplified version of the count_objects function that's failing"""
        def count_objects(image, object_type="bottle"):
            """Count objects in an image using color-based segmentation"""
            # Convert PIL Image to OpenCV format
            if isinstance(image, Image.Image):
                image_cv = np.array(image)
                # Ensure RGB to BGR conversion for OpenCV
                if len(image_cv.shape) == 3 and image_cv.shape[2] == 3:
                    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
            else:
                image_cv = image
            
            # Create grayscale version for processing
            try:
                # THIS IS WHERE THE ERROR OCCURS - need to check channels
                if len(image_cv.shape) == 3:
                    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                else:
                    # Already grayscale
                    gray = image_cv
                
                # Simple dummy count based on image
                count = (np.sum(gray) // 25000) + 1
                return count, image
                
            except Exception as e:
                logger.error(f"Error in count_objects: {e}")
                return 0, image
        
        # Test with RGB image
        rgb_count, _ = count_objects(self.rgb_image)
        self.assertGreater(rgb_count, 0)
        
        # Test with grayscale image
        gray_count, _ = count_objects(self.gray_image)
        self.assertGreater(gray_count, 0)

if __name__ == '__main__':
    unittest.main() 