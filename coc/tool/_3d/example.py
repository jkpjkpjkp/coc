"""
Example usage of the image_to_3d tool for generating 3D representations from images.

This script demonstrates how to use the synergized image-to-3D tool that combines:
- Fast3R for general image-to-3D conversion
- PoinTr for 3D point cloud completion
- TRELLIS for single object image-to-3D
- SAM for object segmentation
- Depth Anything V2 for depth estimation
"""

import os
import argparse
from PIL import Image
from coc.tool import get_image_to_3d

def main():
    """Run the image-to-3D conversion example."""
    parser = argparse.ArgumentParser(description="Generate 3D from image using synergized models")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="./output_3d", 
                        help="Output directory for 3D results")
    parser.add_argument("--use_fast3r", action="store_true", default=True, 
                        help="Use Fast3R for general 3D reconstruction")
    parser.add_argument("--use_pointr", action="store_true", default=True, 
                        help="Use PoinTr for point cloud completion")
    parser.add_argument("--use_trellis", action="store_true", default=True, 
                        help="Use TRELLIS for single object 3D reconstruction")
    parser.add_argument("--use_sam", action="store_true", default=True, 
                        help="Use SAM for object segmentation")
    parser.add_argument("--use_depth", action="store_true", default=True, 
                        help="Use Depth Anything for depth estimation")
    
    args = parser.parse_args()
    
    # Check if input image exists
    if not os.path.exists(args.image):
        print(f"Error: Input image {args.image} does not exist")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load the image
    image = Image.open(args.image).convert("RGB")
    
    print(f"Processing image: {args.image}")
    
    # Create the image-to-3D converter with specified options
    image_to_3d = get_image_to_3d(
        use_fast3r=args.use_fast3r,
        use_pointr=args.use_pointr,
        use_trellis=args.use_trellis,
        use_sam=args.use_sam,
        use_depth=args.use_depth
    )
    
    # Generate 3D representation and save results
    print("Generating 3D representation...")
    results = image_to_3d(image, args.output)
    
    print(f"3D results saved to: {args.output}")
    
    # Print summary of generated outputs
    print("\nGenerated outputs:")
    if "depth_map" in results:
        print("- Depth map")
    if "segmentation" in results:
        print(f"- Segmentation masks ({len(results['segmentation']['masks'])} objects)")
    if "fast3r" in results and "point_cloud" in results["fast3r"]:
        print("- Fast3R point cloud")
    if "trellis" in results:
        print(f"- TRELLIS object models ({len(results['trellis'])} objects)")
    if "final_representation" in results:
        if "point_cloud" in results["final_representation"]:
            print("- Final integrated point cloud")
        if "objects" in results["final_representation"]:
            print(f"- Individual object models ({len(results['final_representation']['objects'])} objects)")

if __name__ == "__main__":
    main() 