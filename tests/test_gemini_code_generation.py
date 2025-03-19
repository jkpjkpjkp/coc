#!/usr/bin/env python3
"""
Test script to analyze and fix the code generation issues in GeminiAgent

This script reproduces the error in code generation and tests various prompt
strategies to improve the generated code's robustness.

not meant to be run by pytest, but rather as a standalone script
"""

import os
import sys
import logging
from pathlib import Path
import tempfile
from PIL import Image
import numpy as np
import cv2

# Add parent directory to path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the GeminiAgent
from coc.tree.gemi import GeminiAgent
from coc.tool.vqa.gemini import Gemini
from coc.util.text import extract_code, extract_boxed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_image(is_grayscale=False):
    """Create a test image for code generation testing"""
    if is_grayscale:
        # Create grayscale test image
        img_array = np.zeros((100, 100), dtype=np.uint8)
        img_array[20:80, 20:80] = 255  # Add a white square
        return Image.fromarray(img_array, mode='L')
    else:
        # Create RGB test image
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img_array[20:80, 20:80, 0] = 255  # Red square
        return Image.fromarray(img_array)

def analyze_generated_code(code, image_type="RGB"):
    """Analyze generated code for potential issues"""
    issues = []
    
    # Check for specific patterns that might cause issues
    if "cv2.cvtColor" in code and "COLOR_BGR2GRAY" in code and image_type == "grayscale":
        issues.append("Converting already grayscale image to grayscale (cv2.COLOR_BGR2GRAY)")
    
    if "cv2.cvtColor" in code and not "len(image.shape)" in code and not "len(image_cv.shape)" in code:
        issues.append("No channel check before color conversion")
    
    return issues

def my_code_generation(input_prompt, image, fix_prompt=False):
    """Test code generation with different prompt strategies"""
    agent = GeminiAgent(use_depth=False, use_segmentation=False)
    
    if fix_prompt:
        # Enhanced prompt with safeguards for image conversion
        enhanced_prompt = f"""
        {input_prompt}
        
        IMPORTANT: The input image could be either RGB or grayscale. Your code MUST check for image type 
        before any color space conversion. Include this pattern in your code:
        
        # Safe grayscale conversion
        if len(img_array.shape) == 3:  # RGB/BGR image
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:  # Already grayscale
            gray = img_array
        """
    else:
        enhanced_prompt = input_prompt
    
    # Generate code using the agent
    response = agent.generate(enhanced_prompt, [image])
    
    # Extract just the code
    code = extract_code(response) or extract_boxed(response) or response
    
    # Analyze the generated code
    image_type = "grayscale" if image.mode == 'L' else "RGB"
    issues = analyze_generated_code(code, image_type)
    
    return code, issues

def test_with_both_image_types(prompt, print_full_code=False):
    """Test the same prompt with both RGB and grayscale images"""
    logger.info("="*80)
    logger.info(f"Testing prompt: {prompt[:100]}...")
    
    # Test with RGB image
    rgb_image = create_test_image(is_grayscale=False)
    rgb_code, rgb_issues = my_code_generation(prompt, rgb_image, fix_prompt=False)
    
    # Test with grayscale image
    gray_image = create_test_image(is_grayscale=True)
    gray_code, gray_issues = my_code_generation(prompt, gray_image, fix_prompt=False)
    
    # Test with fixed prompt for grayscale
    fixed_gray_code, fixed_gray_issues = my_code_generation(prompt, gray_image, fix_prompt=True)
    
    # Print results
    logger.info("RGB image issues: %s", rgb_issues)
    logger.info("Grayscale image issues: %s", gray_issues)
    logger.info("Fixed grayscale issues: %s", fixed_gray_issues)
    
    if print_full_code:
        logger.info("\nGenerated code for RGB image:")
        logger.info(rgb_code[:500] + "..." if len(rgb_code) > 500 else rgb_code)
        
        logger.info("\nGenerated code for grayscale image:")
        logger.info(gray_code[:500] + "..." if len(gray_code) > 500 else gray_code)
        
        logger.info("\nGenerated code with fixed prompt for grayscale:")
        logger.info(fixed_gray_code[:500] + "..." if len(fixed_gray_code) > 500 else fixed_gray_code)
    
    return {
        "rgb_issues": rgb_issues,
        "gray_issues": gray_issues,
        "fixed_issues": fixed_gray_issues
    }

def analyze_prompt_effectiveness():
    """Analyze different prompt strategies for their effectiveness"""
    results = {}
    
    # Test different prompt strategies
    
    # Simple counting prompt
    simple_prompt = "Count the bottles in this image and return the total count."
    results["simple"] = test_with_both_image_types(simple_prompt)
    
    # More detailed prompt with structure
    detailed_prompt = """
    Count the bottles in this image using OpenCV:
    1. Convert the image to grayscale
    2. Apply thresholding to separate bottles from background
    3. Find contours to identify bottle shapes
    4. Count and return the total number of bottles
    """
    results["detailed"] = test_with_both_image_types(detailed_prompt)
    
    # Prompt with explicit safeguards
    safeguard_prompt = """
    Count the bottles in this image using OpenCV.
    
    IMPORTANT: Handle both RGB and grayscale images correctly by checking the input image shape 
    before any color conversion. Implement proper error handling throughout your code.
    """
    results["safeguard"] = test_with_both_image_types(safeguard_prompt)
    
    # Analyze results
    logger.info("\n\nPrompt Effectiveness Summary:")
    for prompt_type, issues in results.items():
        logger.info(f"\n{prompt_type.upper()} PROMPT:")
        logger.info(f"  RGB issues: {len(issues['rgb_issues'])}")
        logger.info(f"  Grayscale issues: {len(issues['gray_issues'])}")
        logger.info(f"  Fixed prompt issues: {len(issues['fixed_issues'])}")
    
    return results

def exec_environment_with_safe_conversion():
    """Test if adding safe_convert_to_gray to the exec environment fixes the issues"""
    from coc.exec.mod import Exec
    
    # Create test images
    rgb_image = create_test_image(is_grayscale=False)
    gray_image = create_test_image(is_grayscale=True)
    
    # Create exec environment
    exec_env = Exec()
    
    # Add safe conversion function
    safe_conversion_code = """
    def safe_convert_to_gray(image):
        '''Safely convert an image to grayscale, handling already-grayscale images'''
        import cv2
        import numpy as np
        
        # Check if image is numpy array
        if not isinstance(image, np.ndarray):
            from PIL import Image
            if isinstance(image, Image.Image):
                image = np.array(image)
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Check if image is already grayscale (2D array)
        if len(image.shape) == 2:
            return image
        # Convert BGR/RGB to grayscale if 3 channels
        elif len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
    """
    
    # Add the images to environment
    exec_env.set_var("rgb_image", rgb_image)
    exec_env.set_var("gray_image", gray_image)
    
    # Execute safe conversion function definition
    stdout, stderr, _ = exec_env._run(safe_conversion_code)
    if stderr:
        logger.error(f"Error defining safe conversion: {stderr}")
        return False
    
    # Test the function with both image types
    test_code = """
    import numpy as np
    import cv2
    
    # Convert PIL to numpy
    rgb_np = np.array(rgb_image)
    gray_np = np.array(gray_image)
    
    # Use safe conversion
    rgb_gray = safe_convert_to_gray(rgb_np)
    gray_gray = safe_convert_to_gray(gray_np)
    
    print(f"RGB image shape: {rgb_np.shape}")
    print(f"Grayscale image shape: {gray_np.shape}")
    print(f"RGB converted to gray shape: {rgb_gray.shape}")
    print(f"Grayscale converted to gray shape: {gray_gray.shape}")
    
    # Test with PIL image directly
    rgb_gray_pil = safe_convert_to_gray(rgb_image)
    gray_gray_pil = safe_convert_to_gray(gray_image)
    print(f"RGB PIL converted to gray shape: {rgb_gray_pil.shape}")
    print(f"Grayscale PIL converted to gray shape: {gray_gray_pil.shape}")
    """
    
    # Execute test code
    stdout, stderr, _ = exec_env._run(test_code)
    if stderr:
        logger.error(f"Error testing safe conversion: {stderr}")
        return False
    
    logger.info("Safe conversion test results:")
    logger.info(stdout)
    
    return "Error" not in stderr

def main():
    """Main function to run all tests"""
    logger.info("Testing GeminiAgent code generation to find and fix grayscale conversion issues")
    
    # Test 1: Analyze prompt effectiveness
    logger.info("\n=== TESTING PROMPT EFFECTIVENESS ===")
    analyze_prompt_effectiveness()
    
    # Test 2: Test exec environment with safe conversion
    logger.info("\n=== TESTING EXEC ENVIRONMENT WITH SAFE CONVERSION ===")
    exec_env_success = exec_environment_with_safe_conversion()
    logger.info(f"Exec environment test {'passed' if exec_env_success else 'failed'}")
    
    # Test 3: Generate a recommended solution
    logger.info("\n=== GENERATING RECOMMENDED SOLUTION ===")
    logger.info("Based on test results, the recommended solution is:")
    logger.info("1. Add a safe_convert_to_gray utility function to the exec environment")
    logger.info("2. Update code generation prompts to explicitly check image channels")
    logger.info("3. Add error handling in count_objects and other image processing functions")

if __name__ == "__main__":
    main() 