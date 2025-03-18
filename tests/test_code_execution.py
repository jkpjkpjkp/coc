"""
Unit tests for the code execution functionality in GeminiAgent

These tests focus on the analyze_with_code_execution method and related
functions to ensure they handle inputs correctly and process Gemini outputs.
"""

import unittest
import os
import sys
import logging
from unittest.mock import MagicMock, patch, call
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from coc.tree.gemi import GeminiAgent
from coc.exec.mod import Exec

class MockExec:
    """Mock Exec class for testing code execution"""
    
    def __init__(self):
        self.globals = {}
        self.run_history = []
    
    def _run(self, code):
        """Mock running code"""
        self.run_history.append(code)
        return "Code execution output", "", []  # stdout, stderr, images
    
    def get_var(self, name):
        """Get a variable from the globals"""
        return self.globals.get(name)
    
    def set_var(self, name, value):
        """Set a variable in globals"""
        self.globals[name] = value


class TestCodeExecution(unittest.TestCase):
    """Test the code execution functionality in GeminiAgent"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test images
        self.test_image = Image.new('RGB', (100, 100), color='red')
        
        # Setup patches
        self.gemini_patcher = patch('coc.tree.gemi.Gemini')
        self.exec_patcher = patch('coc.tree.gemi.Exec', MockExec)
        
        # Start patches
        self.mock_gemini_class = self.gemini_patcher.start()
        self.exec_patcher.start()
        
        # Setup mock gemini
        self.mock_gemini = MagicMock()
        self.mock_gemini_class.return_value = self.mock_gemini
        
        # Create agent and override its native Gemini instance with our mock
        self.agent = GeminiAgent(use_depth=True, use_segmentation=True)
        self.agent.gemini = self.mock_gemini
        
        # Set up agent capabilities
        self.agent.modules_available = {
            "depth": True,
            "segmentation": True,
            "novel_view": False,
            "point_cloud": False
        }
        
        # Patch agent's _get_prompt_capabilities to return a simple string
        self.agent._get_prompt_capabilities = MagicMock(return_value="Test prompt")
        
        # Patch setup_exec_environment to return a MockExec directly
        self.mock_exec = MockExec()
        self.agent.setup_exec_environment = MagicMock(return_value=self.mock_exec)
    
    def tearDown(self):
        """Tear down test fixtures"""
        self.gemini_patcher.stop()
        self.exec_patcher.stop()
    
    def test_analyze_with_code_execution_string_handling(self):
        """Test proper string handling in analyze_with_code_execution"""
        # Setup Gemini responses for the two expected calls
        self.mock_gemini.run_freestyle.side_effect = [
            "```python\nprint('Found 23 bottles')\n```",  # Code generation
            "There are 23 bottles."  # Interpretation
        ]
        
        # Call the method
        result = self.agent.analyze_with_code_execution(
            "How many bottles are there?", 
            [self.test_image]
        )
        
        # Verify the mock was called with the right arguments
        calls = self.mock_gemini.run_freestyle.call_args_list
        
        # The first call should include a string containing the prompt
        self.assertTrue(isinstance(calls[0][0][0], str))
        self.assertIn("How many bottles are there?", calls[0][0][0])
        
        # The second call should be the interpretation prompt
        self.assertTrue(isinstance(calls[1][0][0], str))
        self.assertIn("I ran the image analysis with this code", calls[1][0][0])
        
        # Check that Gemini was called twice
        self.assertEqual(self.mock_gemini.run_freestyle.call_count, 2)
    
    def test_generate_orchestrated_with_counting_prompt(self):
        """Test generate_orchestrated with a counting prompt"""
        # Mock analyze_with_code_execution to ensure it's called
        self.agent.analyze_with_code_execution = MagicMock(return_value="Analysis result")
        
        # Call method
        result = self.agent.generate_orchestrated(
            "Count the bottles on the shelf", 
            [self.test_image]
        )
        
        # Check that analyze_with_code_execution was called
        self.agent.analyze_with_code_execution.assert_called_once_with(
            "Count the bottles on the shelf", 
            [self.test_image]
        )
        
        # Check result
        self.assertEqual(result, "Analysis result")
    
    def test_error_handling_in_code_execution(self):
        """Test error handling in code execution"""
        # Setup Gemini to return valid code
        self.mock_gemini.run_freestyle.return_value = "```python\nprint('test')\n```"
        
        # Setup mock Exec to run code without error
        mock_exec = MockExec()
        
        # Instead of trying to mock _run with exception, which gets complicated,
        # we'll patch the whole Exec._run method to raise an exception when called
        def mock_run_with_error(code):
            raise Exception("Test error")
        
        # Override the _run method
        mock_exec._run = mock_run_with_error
        
        # Replace the agent's setup_exec_environment mock to return our custom mock_exec
        self.agent.setup_exec_environment = MagicMock(return_value=mock_exec)
        
        # Call the method - now should handle the exception
        result = self.agent.analyze_with_code_execution(
            "Analyze this image", 
            [self.test_image]
        )
        
        # Check that the error was handled and returned in the result
        self.assertIn("Error analyzing images with code execution: Test error", result)
    
    def test_debug_and_fix_code(self):
        """Test the code fixing capability"""
        # Setup Gemini responses
        self.mock_gemini.run_freestyle.side_effect = [
            "```python\nprint('test')\n```",  # Original code
            "```python\nprint('fixed code')\n```",  # Fixed code
            "Final interpretation"  # Interpretation
        ]
        
        # Create a custom mock_exec that will return an error on first run
        mock_exec = MockExec()
        
        # Set up a counter to track calls
        run_count = [0]
        
        # Define custom _run method with side effects
        def custom_run(code):
            run_count[0] += 1
            if run_count[0] == 1:
                # First call - return error
                return "", "Error in code", []
            else:
                # Subsequent calls - return success
                return "Fixed code output", "", []
        
        # Override the _run method
        mock_exec._run = custom_run
        
        # Replace the agent's setup_exec_environment mock
        self.agent.setup_exec_environment = MagicMock(return_value=mock_exec)
        
        # Call the method
        result = self.agent.analyze_with_code_execution(
            "Fix this code", 
            [self.test_image]
        )
        
        # Check that Gemini was called three times:
        # 1. Generate initial code
        # 2. Fix the code
        # 3. Interpret the results
        self.assertEqual(self.mock_gemini.run_freestyle.call_count, 3)
        
        # Check result contains both the fixed output and interpretation
        self.assertIn("Fixed code output", result)
        self.assertIn("Final interpretation", result)
    
    def test_extraction_of_function_interfaces(self):
        """Test extraction of function interfaces"""
        # Create a prompt that includes the function interfaces section
        mock_prompt = """
        You are a powerful AI.
        
        Available vision tool interfaces:
        ```python
        def test_function(): ...
        ```
        
        For vision tasks:
        Do something.
        """
        
        # Override the mock for _get_prompt_capabilities 
        self.agent._get_prompt_capabilities = MagicMock(return_value=mock_prompt)
        
        # Setup minimal Gemini response to avoid errors
        self.mock_gemini.run_freestyle.side_effect = [
            "```python\nprint('test')\n```",
            "Test interpretation"
        ]
        
        # Call the method
        self.agent.analyze_with_code_execution("Test prompt", [self.test_image])
        
        # Check the first call to run_freestyle
        first_call_args = self.mock_gemini.run_freestyle.call_args_list[0][0][0]
        self.assertIn("def test_function()", first_call_args)


if __name__ == "__main__":
    unittest.main() 