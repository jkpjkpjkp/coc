#!/usr/bin/env python3
"""
Test script for the retry mechanism in Gemini class

This script verifies that the retry mechanism properly handles API errors
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, call, PropertyMock
import time
import logging

# Add parent directory to path to import coc modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to test
from coc.tool.vqa.gemini import Gemini, retry_on_api_error, RETRY_ERRORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockChatOpenAI:
    """Mock class for ChatOpenAI that can be properly serialized by Pydantic"""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        
    def invoke(self, *args, **kwargs):
        return MagicMock()

class TestGeminiRetry(unittest.TestCase):
    """Test cases for the retry mechanism in Gemini class"""
    
    def setUp(self):
        """Set up test environment"""
        # Make sure required env vars are set for testing
        if 'GEMINI_API_KEY' not in os.environ:
            os.environ['GEMINI_API_KEY'] = 'test_api_key'
        if 'GEMINI_BASE_URL' not in os.environ:
            os.environ['GEMINI_BASE_URL'] = 'https://test.example.com'
        
        # Patch ChatOpenAI to avoid Pydantic validation errors
        self.chat_openai_patcher = patch('coc.tool.vqa.gemini.ChatOpenAI', MockChatOpenAI)
        self.mock_chat_openai = self.chat_openai_patcher.start()
        
        # Create test instance with short retry settings for faster tests
        self.gemini = Gemini(max_retries=3, base_delay=0.1, max_delay=0.5, backoff_factor=1.5)
    
    def tearDown(self):
        """Clean up after tests"""
        self.chat_openai_patcher.stop()
    
    def test_initialization_with_custom_retry_values(self):
        """Test that custom retry values are properly set"""
        gemini = Gemini(max_retries=10, base_delay=2.0, max_delay=60.0, backoff_factor=3.0)
        
        # Use object.__getattribute__ to safely access attributes in a Pydantic model
        self.assertEqual(object.__getattribute__(gemini, 'max_retries'), 10)
        self.assertEqual(object.__getattribute__(gemini, 'base_delay'), 2.0)
        self.assertEqual(object.__getattribute__(gemini, 'max_delay'), 60.0)
        self.assertEqual(object.__getattribute__(gemini, 'backoff_factor'), 3.0)
    
    @patch('time.sleep')
    def test_retry_on_api_error_decorator(self, mock_sleep):
        """Test that the retry_on_api_error decorator works properly"""
        # Define a test function that will be decorated
        retry_count = [0]  # Use a list to store mutable state
        
        @retry_on_api_error(max_retries=3, base_delay=0.1, max_delay=0.5)
        def test_func():
            retry_count[0] += 1
            if retry_count[0] < 3:
                error_class = ConnectionError if retry_count[0] == 1 else Exception
                error_msg = "Connection refused" if retry_count[0] == 1 else "502 Bad Gateway"
                raise error_class(error_msg)
            return "success"
            
        # Call the decorated function
        result = test_func()
        
        # Verify the function was retried the correct number of times
        self.assertEqual(retry_count[0], 3)
        
        # Verify the result
        self.assertEqual(result, "success")
        
        # Verify sleep was called with increasing delays
        self.assertEqual(mock_sleep.call_count, 2)  # Should be called after first and second errors
    
    @patch('time.sleep')
    def test_retry_with_non_retryable_error(self, mock_sleep):
        """Test that non-retryable errors are not retried"""
        # Define a test function that will be decorated
        retry_count = [0]
        
        @retry_on_api_error(max_retries=3, base_delay=0.1)
        def test_func():
            retry_count[0] += 1
            raise ValueError("Invalid input parameter")
            
        # Call the decorated function - should raise the error
        with self.assertRaises(ValueError):
            test_func()
        
        # Verify the function was called only once
        self.assertEqual(retry_count[0], 1)
        
        # Verify sleep was not called
        mock_sleep.assert_not_called()
    
    @patch('time.sleep')
    def test_retry_with_backoff_method(self, mock_sleep):
        """Test the _retry_with_backoff method in Gemini class"""
        # Create a counter for the number of calls
        call_count = [0]
        
        # Create a test function that raises errors for the first few calls
        def test_func(*args, **kwargs):
            call_count[0] += 1
            
            if call_count[0] == 1:
                raise ConnectionError("Connection refused")
            elif call_count[0] == 2:
                raise Exception("500 Internal Server Error")
            elif call_count[0] == 3:
                raise Exception("API rate limit exceeded")
            return "success"
        
        # Call _retry_with_backoff
        result = self.gemini._retry_with_backoff(test_func, "test", param="value")
        
        # Verify the function was called the correct number of times
        self.assertEqual(call_count[0], 4)
        
        # Verify result
        self.assertEqual(result, "success")
        
        # Verify sleep was called the right number of times
        self.assertEqual(mock_sleep.call_count, 3)  # Should be called after each of the three errors
    
    def test_run_freestyle_with_retry(self):
        """Test that run_freestyle uses the retry mechanism"""
        # Create a wrapper for _retry_with_backoff that we can mock
        original_retry = self.gemini._retry_with_backoff
        
        try:
            # Replace _retry_with_backoff with a mock
            mock_retry = MagicMock(return_value="test response")
            self.gemini._retry_with_backoff = mock_retry
            
            # Call run_freestyle
            result = self.gemini.run_freestyle(["test prompt"])
            
            # Verify _retry_with_backoff was called
            mock_retry.assert_called_once()
            
            # Verify first argument is the implementation function
            self.assertEqual(mock_retry.call_args[0][0], self.gemini._run_freestyle_impl)
            
            # Verify second argument is the prompt
            self.assertEqual(mock_retry.call_args[0][1], ["test prompt"])
            
            # Verify result
            self.assertEqual(result, "test response")
            
        finally:
            # Restore original method
            self.gemini._retry_with_backoff = original_retry
    
    def test_gemini_as_llm_helper(self):
        """Test that gemini_as_llm helper functions correctly"""
        # Import helper function
        from coc.tool.vqa.gemini import gemini_as_llm
        
        # Replace Gemini class with a mock that returns a mock instance
        with patch('coc.tool.vqa.gemini.Gemini') as mock_gemini_class:
            # Create mock instance
            mock_gemini_instance = MagicMock()
            mock_gemini_class.return_value = mock_gemini_instance
            
            # Setup mock for _retry_with_backoff
            mock_retry = MagicMock(return_value="test response")
            mock_gemini_instance._retry_with_backoff = mock_retry
            
            # Call the helper to get an LLM function
            llm_func = gemini_as_llm(max_retries=2, base_delay=0.2)
            
            # Call the LLM function
            result = llm_func("test query")
            
            # Verify _retry_with_backoff was called with invoke and the query
            mock_retry.assert_called_once()
            self.assertEqual(mock_retry.call_args[0][0], mock_gemini_instance.invoke)
            self.assertEqual(mock_retry.call_args[0][1], "test query")
            
            # Verify result
            self.assertEqual(result, "test response")

if __name__ == "__main__":
    unittest.main() 