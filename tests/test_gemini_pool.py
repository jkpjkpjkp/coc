#!/usr/bin/env python3

import os
import logging
from PIL import Image
import numpy as np
import time
import random

from coc.tool.vqa.gemini import Gemini, GEMINI_BROKERS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_image(width=512, height=512):
    """Create a simple test image with some shapes for VQA testing"""
    # Create white background
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw a red circle
    center = (width // 4, height // 4)
    radius = min(width, height) // 8
    for y in range(height):
        for x in range(width):
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 < radius ** 2:
                img[y, x] = [255, 0, 0]  # Red
    
    # Draw a blue square
    top_left = (width // 2, height // 2)
    size = min(width, height) // 4
    img[top_left[1]:top_left[1]+size, top_left[0]:top_left[0]+size] = [0, 0, 255]  # Blue
    
    return Image.fromarray(img)

def test_gemini_broker_pool():
    """Test the Gemini broker pool implementation"""
    logger.info("Testing Gemini broker pool with all available brokers...")
    logger.info(f"Available brokers: {len(GEMINI_BROKERS)}")
    
    # Display broker information
    for i, broker in enumerate(GEMINI_BROKERS):
        api_key, base_url = broker
        # Only show first few chars of API key for security
        masked_key = api_key[:8] + "..." + api_key[-4:]
        logger.info(f"Broker {i+1}: URL={base_url}, API Key={masked_key}")
    
    # Define a custom broker pool for testing
    # Let's add an intentionally invalid broker to test failover
    test_brokers = GEMINI_BROKERS.copy()
    test_brokers.append(["invalid-api-key", "https://non-existent-url.example.com"])
    
    # Create an instance with the broker pool
    gemini = Gemini(
        max_retries=2,  # Keep retries low for testing
        base_delay=1.0,
        max_delay=5.0,
        backoff_factor=1.5,
        broker_pool=test_brokers
    )
    
    # Create a test image
    test_image = create_test_image()
    
    # Simple VQA question
    question = "What shapes can you see in this image?"
    
    # Test broker failover
    def test_broker_failover():
        # Force an invalid broker first
        gemini.current_broker_index = len(test_brokers) - 1  # Set to the invalid broker
        gemini._update_env_from_broker_pool()  # Update env vars
        gemini.mllm = gemini._create_mllm_instance()  # Recreate MLLM
        
        logger.info(f"Starting with invalid broker (index {gemini.current_broker_index})")
        
        try:
            # This should fail and trigger broker switching
            response = gemini._run(test_image, question)
            logger.info(f"Response received (unexpected success): {response}")
            return True
        except Exception as e:
            # We expect this to fail and switch brokers
            logger.info(f"Expected error with invalid broker: {str(e)}")
            return False
    
    # Reset to first broker for normal test
    gemini.current_broker_index = 0
    gemini._update_env_from_broker_pool()
    gemini.mllm = gemini._create_mllm_instance()
    
    try:
        # Run basic VQA with the broker pool
        logger.info(f"Running VQA with question: '{question}'")
        
        # Add a small delay to prevent rate limiting
        time.sleep(random.uniform(1.0, 2.0))
        
        response = gemini._run(test_image, question)
        logger.info(f"Response received: {response}")
        logger.info(f"Test completed successfully using broker index {gemini.current_broker_index}")
        
        # Optionally test broker failover
        # Uncomment the next line to test broker failover
        # test_broker_failover()
        
        return True
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_gemini_broker_pool()
    if success:
        print("✅ Test completed successfully")
    else:
        print("❌ Test failed") 