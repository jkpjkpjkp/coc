import argparse
from PIL import Image
import os
from coc.tree.gemi import GeminiAgent, main_eval_muir

def test_agent_with_image(image_path, prompt=None, use_orchestration=True, verbose=False):
    """Test the Gemini agent with a single image"""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        print(f"Successfully loaded image: {image_path} ({image.width}x{image.height})")
        
        # Create agent with appropriate settings
        agent = GeminiAgent(
            use_depth=False,  # Set to True to enable depth estimation if available
            use_segmentation=True,  # Enable segmentation for counting
            use_novel_view=False,  # Set to True to enable novel view synthesis if available
            use_point_cloud=False,  # Set to True to enable point cloud processing if available
            verbose=verbose  # Enable verbose logging if requested
        )
        
        # Default prompt if none provided
        if prompt is None:
            prompt = "Describe what you see in this image, focusing on the content and relevant details."
        
        # Run agent
        print(f"Processing image: {image_path}")
        print(f"Prompt: {prompt}")
        print(f"Using {'orchestrated' if use_orchestration else 'standard'} approach")
        print("Generating response...")
        
        try:
            # Use either orchestrated or standard approach
            if use_orchestration:
                response = agent.generate_orchestrated(prompt, [image])
            else:
                response = agent.generate(prompt, [image])
            
            print("\nResponse:")
            print("="*50)
            print(response)
            print("="*50)
            
            if response.startswith("Error:"):
                print("\nEncountered an error in the response. Check API settings and connectivity.")
            
            # Print tool usage statistics if verbose
            if verbose and use_orchestration:
                print("\nTool usage statistics:")
                for tool, count in agent.tool_call_counts.items():
                    if count > 0:
                        print(f"- {tool}: {count} calls")
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

def run_muir_evaluation(partition):
    """Run evaluation on MUIR dataset"""
    print(f"Running evaluation on MUIR {partition} partition...")
    success_rate = main_eval_muir(partition)
    return success_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Gemini 3D Agent")
    parser.add_argument("--image", type=str, help="Path to image for testing")
    parser.add_argument("--prompt", type=str, help="Custom prompt for image analysis")
    parser.add_argument("--eval", choices=["Counting", "Ordering"], 
                        help="Run evaluation on MUIR dataset with specified partition")
    parser.add_argument("--standard", action="store_true", 
                        help="Use standard approach instead of orchestrated approach")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.image:
        test_agent_with_image(args.image, args.prompt, not args.standard, args.verbose)
    elif args.eval:
        run_muir_evaluation(args.eval)
    else:
        print("Please provide either --image or --eval argument")
        print("Example usage:")
        print("  python test_gemi.py --image path/to/image.jpg")
        print("  python test_gemi.py --image path/to/image.jpg --prompt 'Describe this scene'")
        print("  python test_gemi.py --image path/to/image.jpg --prompt 'How many bottles?' --verbose")
        print("  python test_gemi.py --eval Counting") 