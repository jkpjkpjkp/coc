#!/usr/bin/env python3
"""
MUIR Dataset Evaluation Script

This script evaluates the GeminiAgent on the MUIR dataset for
tasks like counting and ordering.
"""

import argparse
import os
import logging
from coc.tree.gemi import GeminiAgent, main_eval_muir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("muir-eval")

def run_muir_evaluation(partition, use_3d_vision=False):
    """
    Run evaluation on MUIR dataset
    
    Args:
        partition: Which partition to evaluate ('Counting' or 'Ordering')
        use_3d_vision: Whether to use 3D vision enhancements (if available)
    """
    logger.info(f"Running evaluation on MUIR {partition} partition...")
    logger.info(f"3D vision enhancements: {'Enabled' if use_3d_vision else 'Disabled'}")
    
    try:
        # Initialize the agent with or without 3D vision capabilities
        agent = GeminiAgent(
            use_depth=use_3d_vision,
            use_segmentation=use_3d_vision,
            use_novel_view=use_3d_vision,
            use_point_cloud=use_3d_vision
        )
        
        # Get available capabilities
        capabilities = []
        if agent.use_depth:
            capabilities.append("depth estimation")
        if agent.use_segmentation:
            capabilities.append("segmentation")
        if agent.use_novel_view:
            capabilities.append("novel view synthesis")
        if agent.use_point_cloud:
            capabilities.append("point cloud processing")
        
        if capabilities:
            logger.info(f"Active capabilities: {', '.join(capabilities)}")
        else:
            logger.info("No 3D vision capabilities active")
        
        # Import here to avoid circular imports
        from coc.data.muir import muir, LoadMuir
        
        # Check if dataset exists
        try:
            dataset = LoadMuir(partition)
            logger.info(f"Successfully loaded MUIR {partition} dataset")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return False
        
        # Load dataset and evaluate
        batch = muir(partition)
        
        # Convert to list to check length
        tasks = list(batch)
        logger.info(f"Dataset contains {len(tasks)} tasks")
        
        # Run evaluation
        from coc.tree.gemi import eval_a_batch
        correct, total = eval_a_batch(tasks, agent)
        
        success_rate = correct / total if total > 0 else 0
        result_text = f"MUIR {partition} evaluation results: {success_rate:.2%} ({correct}/{total})"
        logger.info(result_text)
        
        # Save results to file
        results_file = f"muir_{partition.lower()}_results.txt"
        with open(results_file, 'w') as f:
            f.write(f"MUIR {partition} Evaluation Results\n")
            f.write(f"================================\n\n")
            f.write(f"3D Vision: {'Enabled' if use_3d_vision else 'Disabled'}\n")
            if capabilities:
                f.write(f"Active capabilities: {', '.join(capabilities)}\n")
            f.write(f"\nSuccess rate: {success_rate:.2%} ({correct}/{total})\n")
        
        logger.info(f"Results saved to {results_file}")
        return success_rate
    
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate GeminiAgent on MUIR dataset")
    parser.add_argument("partition", choices=["Counting", "Ordering"], 
                        help="MUIR partition to evaluate")
    parser.add_argument("--use-3d", action="store_true", 
                        help="Enable 3D vision enhancements if available")
    
    args = parser.parse_args()
    
    # Run evaluation
    run_muir_evaluation(args.partition, args.use_3d)

if __name__ == "__main__":
    main() 