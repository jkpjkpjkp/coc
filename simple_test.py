#!/usr/bin/env python3
"""
Simple test script for Gemini VQA.

This script directly uses the Gemini class from coc.tool.vqa.gemini
to process an image.
"""

import os
import sys
import argparse
from PIL import Image
from coc.tool.vqa.gemini import Gemini

def process_image_with_gemini(image_path, prompt):
    """
    Process an image with Gemini VQA tool
    
    Args:
        image_path: Path to the image
        prompt: Text prompt for the model
    """
    # Load the image
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            print(f"Image loaded: {image_path} ({img.width}x{img.height})")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Create Gemini instance
    try:
        gemini = Gemini()
    except Exception as e:
        print(f"Error creating Gemini instance: {e}")
        print("Make sure environment variables GEMINI_API_KEY and GEMINI_BASE_URL are set")
        return None
    
    # Process the image
    print("Processing image with Gemini...")
    try:
        # Using run_freestyle to handle both text and image
        inputs = [prompt, img]
        response = gemini.run_freestyle(inputs)
        return response
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Test Gemini Vision API")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--prompt", default="Describe what you see in this image", 
                        help="Prompt for the model")
    
    args = parser.parse_args()
    
    # Process the image
    result = process_image_with_gemini(args.image, args.prompt)
    
    if result:
        print("\nResponse:")
        print("="*50)
        print(result)
        print("="*50)
    else:
        print("Failed to get a response from Gemini.")

if __name__ == "__main__":
    main() 