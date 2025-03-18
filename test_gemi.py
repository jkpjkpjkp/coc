#!/usr/bin/env python3
"""
Test script for the GeminiAgent with real images

This script loads a test image and runs the GeminiAgent on it to verify 
that it works correctly after fixes.
"""

import os
import sys
import argparse
import logging
from PIL import Image
import traceback

# Import GeminiAgent
from coc.tree.gemi import GeminiAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test-gemi")

def test_agent_with_image(image_path, prompt, use_orchestration=True):
    """
    Test GeminiAgent with an image and prompt
    
    Args:
        image_path: Path to test image
        prompt: Text prompt to test with
        use_orchestration: Whether to use the orchestrated approach
    """
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        logger.info(f"Successfully loaded image: {image_path} ({image.width}x{image.height})")
        
        # Initialize agent
        agent = GeminiAgent(
            use_depth=True,
            use_segmentation=True,
            use_novel_view=False,
            use_point_cloud=False,
            verbose=True
        )
        
        # Process image
        logger.info(f"Processing image: {image_path}")
        logger.info(f"Prompt: {prompt}")
        
        # Generate response
        if use_orchestration:
            logger.info("Using orchestrated approach")
            logger.info("Generating response...")
            try:
                response = agent.generate_orchestrated(prompt, [image])
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                traceback.print_exc()
                return
        else:
            logger.info("Using direct approach")
            logger.info("Generating response...")
            try:
                response = agent.generate(prompt, [image])
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                traceback.print_exc()
                return
                
        # Print response
        logger.info("Response from GeminiAgent:")
        print("\n" + "="*50)
        print(response)
        print("="*50 + "\n")
        
        # Print tool usage
        used_tools = {k: v for k, v in agent.tool_call_counts.items() if v > 0}
        if used_tools:
            logger.info("Tools used:")
            for tool, count in used_tools.items():
                logger.info(f"- {tool}: {count} calls")
        
    except Exception as e:
        logger.error(f"Error in test_agent_with_image: {str(e)}")
        traceback.print_exc()

def main():
    """Parse arguments and run test"""
    parser = argparse.ArgumentParser(description="Test GeminiAgent with an image")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--prompt", default="Describe this image in detail", help="Prompt to test with")
    parser.add_argument("--direct", action="store_true", help="Use direct approach instead of orchestrated")
    
    args = parser.parse_args()
    
    # Validate image path
    if not os.path.exists(args.image):
        logger.error(f"Image file not found: {args.image}")
        sys.exit(1)
    
    # Run test
    test_agent_with_image(args.image, args.prompt, not args.direct)

if __name__ == "__main__":
    main() 