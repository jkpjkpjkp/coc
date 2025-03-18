#!/usr/bin/env python3
"""
Gemini 3D Agent Runner

This script provides a command-line interface to run the GeminiAgent with enhanced 3D 
vision capabilities for various tasks.
"""

import argparse
import os
import sys
from pathlib import Path
import logging

from coc.tree.gemi import GeminiAgent, main_eval_muir
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("gemi-runner")

def process_image(image_path, prompt=None, save_response=False, save_path=None):
    """
    Process a single image with the GeminiAgent
    
    Args:
        image_path: Path to the image file
        prompt: Custom prompt to use (defaults to general 3D analysis)
        save_response: Whether to save the response to a file
        save_path: Path to save the response (defaults to image_path + .response.txt)
    """
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return False
    
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        logger.info(f"Loaded image: {image_path} ({image.width}x{image.height})")
        
        # Create agent with all 3D capabilities
        agent = GeminiAgent(
            use_depth=True, 
            use_segmentation=True,
            use_novel_view=True,
            use_point_cloud=True
        )
        
        # Default prompt for 3D analysis
        if prompt is None:
            prompt = (
                "Analyze this image in detail. Describe the scene, focusing on:"
                "\n1. The 3D spatial arrangement of objects"
                "\n2. Depth relationships between elements"
                "\n3. Potential occlusions and hidden structures"
                "\n4. How the scene would look from different viewpoints"
            )
        
        # Process the image
        logger.info("Generating response from Gemini with 3D vision enhancements...")
        response = agent.generate(prompt, [image])
        
        # Display response
        print("\n" + "="*50)
        print("RESPONSE:")
        print(response)
        print("="*50 + "\n")
        
        # Save response if requested
        if save_response:
            save_path = save_path or f"{image_path}.response.txt"
            with open(save_path, 'w') as f:
                f.write(response)
            logger.info(f"Response saved to {save_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return False

def process_batch(directory, prompt=None, save_responses=False):
    """
    Process all images in a directory
    
    Args:
        directory: Directory containing images
        prompt: Custom prompt to use for all images
        save_responses: Whether to save responses to files
    """
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        return False
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(directory).glob(f"*{ext}")))
        image_files.extend(list(Path(directory).glob(f"*{ext.upper()}")))
    
    if not image_files:
        logger.error(f"No image files found in {directory}")
        return False
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process each image
    success_count = 0
    for img_path in image_files:
        logger.info(f"Processing {img_path.name}...")
        if process_image(str(img_path), prompt, save_responses):
            success_count += 1
    
    logger.info(f"Successfully processed {success_count}/{len(image_files)} images")
    return success_count > 0

def run_eval(partition, save_results=False):
    """
    Run evaluation on MUIR dataset
    
    Args:
        partition: Which partition to evaluate ('Counting' or 'Ordering')
        save_results: Whether to save results to a file
    """
    logger.info(f"Running evaluation on MUIR {partition} partition...")
    
    try:
        success_rate = main_eval_muir(partition)
        
        result_text = f"MUIR {partition} evaluation results: {success_rate:.2%} success rate"
        logger.info(result_text)
        
        if save_results:
            results_file = f"muir_{partition.lower()}_results.txt"
            with open(results_file, 'w') as f:
                f.write(result_text + "\n")
            logger.info(f"Results saved to {results_file}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run Gemini 3D Agent for image analysis and evaluation"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Image processing command
    image_parser = subparsers.add_parser("image", help="Process a single image")
    image_parser.add_argument("path", help="Path to image file")
    image_parser.add_argument("--prompt", help="Custom prompt for image analysis")
    image_parser.add_argument("--save", action="store_true", help="Save response to a file")
    image_parser.add_argument("--output", help="Path to save response (default: image path + .response.txt)")
    
    # Batch processing command
    batch_parser = subparsers.add_parser("batch", help="Process all images in a directory")
    batch_parser.add_argument("directory", help="Directory containing images")
    batch_parser.add_argument("--prompt", help="Custom prompt for all images")
    batch_parser.add_argument("--save", action="store_true", help="Save responses to files")
    
    # Evaluation command
    eval_parser = subparsers.add_parser("eval", help="Run evaluation on MUIR dataset")
    eval_parser.add_argument("partition", choices=["Counting", "Ordering"], 
                             help="MUIR partition to evaluate")
    eval_parser.add_argument("--save", action="store_true", help="Save results to a file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run appropriate command
    if args.command == "image":
        process_image(args.path, args.prompt, args.save, args.output)
    elif args.command == "batch":
        process_batch(args.directory, args.prompt, args.save)
    elif args.command == "eval":
        run_eval(args.partition, args.save)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 