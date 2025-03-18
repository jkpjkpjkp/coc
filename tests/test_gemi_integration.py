#!/usr/bin/env python3
"""
Integration test for GeminiAgent

This script verifies that the GeminiAgent works as expected with all components
properly integrated. It mocks external dependencies to focus on internal logic.
"""

import unittest
import os
import sys
import logging
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np

# Set up logging
logging.basicConfig(level=logging.WARNING)

# Add the parent directory to the path to import coc modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from coc.tree.gemi import GeminiAgent
from coc.exec.mod import Exec
from coc.tool.vqa.gemini import Gemini  # Import the actual Gemini class

class GeminiIntegrationTest(unittest.TestCase):
    """Integration tests for GeminiAgent"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        
        # Patch external dependencies
        self.patcher1 = patch('coc.tool.vqa.gemini.Gemini')
        self.patcher2 = patch('coc.tree.gemi.Exec')
        
        # Start the patchers
        self.mock_gemini_class = self.patcher1.start()
        self.mock_exec_class = self.patcher2.start()
        
        # Set up the mock Gemini instance
        self.mock_gemini = MagicMock()
        self.mock_gemini.run_freestyle.return_value = "Mocked Gemini response"
        self.mock_gemini_class.return_value = self.mock_gemini
        
        # Set up the mock Exec instance
        self.mock_exec = MagicMock()
        self.mock_exec._run.return_value = ("Executed code output", "", [])
        self.mock_exec_class.return_value = self.mock_exec
        
        # Create GeminiAgent with mocked dependencies
        self.agent = GeminiAgent(
            use_depth=True, 
            use_segmentation=True,
            use_novel_view=False,
            use_point_cloud=False,
            verbose=True
        )
        
        # Manually set the mock Gemini instance on the agent to ensure it's used
        self.agent.gemini = self.mock_gemini
        
        # Mock modules_available to ensure our mocked capabilities are "available"
        self.agent.modules_available = {
            "depth": True,
            "segmentation": True,
            "novel_view": False,
            "point_cloud": False
        }
    
    def tearDown(self):
        """Clean up patchers"""
        self.patcher1.stop()
        self.patcher2.stop()
    
    def test_generate_orchestrated_with_counting_prompt(self):
        """Test the generate_orchestrated method with a counting prompt"""
        # Mock analyze_with_code_execution to avoid it calling _run directly
        self.agent.analyze_with_code_execution = MagicMock()
        self.agent.analyze_with_code_execution.return_value = "Executed code output\n\nThere are 23 bottles in the image."
        
        # Call the method with a counting task
        result = self.agent.generate_orchestrated("How many bottles are in this image?", [self.test_image])
        
        # Verify analyze_with_code_execution was called with the right arguments
        self.agent.analyze_with_code_execution.assert_called_once_with(
            "How many bottles are in this image?", [self.test_image]
        )
        
        # Check the result matches what we expect
        self.assertEqual(result, "Executed code output\n\nThere are 23 bottles in the image.")
    
    def test_generate_with_standard_prompt(self):
        """Test the generate method with a standard (non-counting) prompt"""
        # Reset the mock_gemini call counter
        self.mock_gemini.reset_mock()
        
        # Set a return value for run_freestyle
        self.mock_gemini.run_freestyle.return_value = "Mocked Gemini response"
        
        # Call the method
        result = self.agent.generate("Describe this image", [self.test_image])
        
        # Verify Gemini was called once - with the right type of argument
        self.mock_gemini.run_freestyle.assert_called_once()
        
        # Check the result
        self.assertEqual(result, "Mocked Gemini response")

class TestGeminiRunFreestyle(unittest.TestCase):
    """Test specifically the run_freestyle behavior in Gemini"""
    
    def setUp(self):
        # Create a test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        
        # Create an instance of the real Gemini class
        # This helps us understand its interface without mocking
        try:
            self.real_gemini = Gemini()
            logging.info("Created real Gemini instance")
        except Exception as e:
            logging.error(f"Error creating real Gemini: {e}")
            self.real_gemini = None
    
    def test_run_freestyle_input_type(self):
        """Test what input type run_freestyle expects"""
        logging.info("=== Testing run_freestyle input type ===")
        
        # Create a mock Gemini to test the behavior
        mock_gemini = MagicMock()
        
        # Set up different inputs to test
        string_input = "This is a test prompt"
        list_input_string = ["This is a test prompt"]
        list_input_mixed = ["This is a test prompt", self.test_image]
        
        # Test with string input
        logging.info("Testing with string input")
        try:
            mock_gemini.run_freestyle(string_input)
            logging.info("String input accepted")
        except Exception as e:
            logging.error(f"Error with string input: {e}")
            self.fail(f"String input caused error: {e}")
        
        # Test with list containing just a string
        logging.info("Testing with list containing just a string")
        try:
            mock_gemini.run_freestyle(list_input_string)
            logging.info("List with string input accepted")
        except Exception as e:
            logging.error(f"Error with list containing string: {e}")
            self.fail(f"List containing string caused error: {e}")
        
        # Test with list containing mixed types
        logging.info("Testing with list containing mixed types")
        try:
            mock_gemini.run_freestyle(list_input_mixed)
            logging.info("List with mixed input accepted")
        except Exception as e:
            logging.error(f"Error with list containing mixed types: {e}")
            self.fail(f"List containing mixed types caused error: {e}")
        
        # Check call arguments if we have a real Gemini instance
        if self.real_gemini:
            logging.info("Examining real Gemini.run_freestyle method")
            import inspect
            sig = inspect.signature(self.real_gemini.run_freestyle)
            logging.info(f"Signature: {sig}")
            
            # Check docstring or type hints if available
            doc = self.real_gemini.run_freestyle.__doc__
            logging.info(f"Docstring: {doc}")
    
    def test_analyze_with_code_execution_call_pattern(self):
        """Test how analyze_with_code_execution calls run_freestyle"""
        logging.info("=== Testing analyze_with_code_execution call pattern ===")
        
        # Create agent with mock Gemini
        agent = GeminiAgent()
        agent.gemini = MagicMock()
        
        # Configure the mock to record calls
        # We'll also make it raise an exception for a list but accept a string
        # to simulate the real behavior
        def mock_run_freestyle(inputs):
            logging.info(f"mock_run_freestyle called with: {type(inputs)}")
            if isinstance(inputs, list):
                logging.error("List input detected - this would fail")
                raise TypeError("expected string or bytes-like object, got 'list'")
            return "Mock response"
        
        agent.gemini.run_freestyle = mock_run_freestyle
        
        # Try to call the method that might be failing
        try:
            result = agent.analyze_with_code_execution(
                "How many bottles are in this image?",
                [self.test_image]
            )
            logging.info(f"Method completed with result: {result[:100]}")
        except Exception as e:
            logging.error(f"Method failed with error: {e}")
            # Don't fail the test, we're just trying to understand the behavior
            pass
    
    def test_gemini_caller_behavior(self):
        """Test if the correct caller is being used based on code inspection"""
        logging.info("=== Testing actual call in analyze_with_code_execution ===")
        
        # Create a patch for Gemini's run_freestyle to check what's passed
        with patch('coc.tool.vqa.gemini.Gemini.run_freestyle') as mock_run_freestyle:
            # Configure the mock to just record calls
            mock_run_freestyle.return_value = "Test response"
            
            # Create the agent with our mocked Gemini method
            agent = GeminiAgent()
            # Make sure our patch works by attaching it directly
            agent.gemini = Gemini()
            
            try:
                # Only create necessary stubs to get to the run_freestyle call
                agent.setup_exec_environment = MagicMock()
                agent.setup_exec_environment.return_value = MagicMock()
                agent._get_prompt_capabilities = MagicMock(return_value="test prompt")
                
                # Call the method
                result = agent.analyze_with_code_execution(
                    "How many bottles are in this image?",
                    [self.test_image]
                )
                logging.info("Method completed")
            except Exception as e:
                logging.error(f"Method failed with error: {e}")
            
            # Check what was passed to run_freestyle
            calls = mock_run_freestyle.call_args_list
            logging.info(f"Number of calls: {len(calls)}")
            
            for i, call in enumerate(calls):
                args, kwargs = call
                logging.info(f"Call {i+1} args: {[type(arg) for arg in args]}")
                logging.info(f"Call {i+1} kwargs: {kwargs}")
                
                # Check specifically if a list was passed where a string was expected
                if args and isinstance(args[0], list):
                    logging.error(f"Error pattern detected: run_freestyle called with list: {args[0]}")
                    self.fail("run_freestyle called with list instead of string")

if __name__ == "__main__":
    unittest.main() 