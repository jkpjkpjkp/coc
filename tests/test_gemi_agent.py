import unittest
import os
import sys
import logging
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np
import io

# Add the parent directory to the path to import coc modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from coc.tree.gemi import GeminiAgent, generate_one_child as gemi_generate_one_child, generate_children as gemi_generate_children
from coc.tree.une import TreeNode

# Set up logging and suppress logs during tests
logging.basicConfig(level=logging.ERROR)

class MockExec:
    """Mock Exec class for testing"""
    def __init__(self):
        self.globals = {}
        self.run_history = []
    
    def _run(self, code):
        self.run_history.append(code)
        return "Executed code", "", []  # stdout, stderr, images
    
    def get_var(self, name):
        return self.globals.get(name)
    
    def set_var(self, name, value):
        self.globals[name] = value


class TestGeminiAgentComponents(unittest.TestCase):
    """Test individual components of the GeminiAgent class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        
        # Mock Gemini for testing
        self.mock_gemini = MagicMock()
        self.mock_gemini.run_freestyle.return_value = "Test response"
        
        # Create a GeminiAgent with mocked Gemini
        self.agent = GeminiAgent(
            use_depth=False,
            use_segmentation=False,
            use_novel_view=False,
            use_point_cloud=False,
            verbose=True
        )
        self.agent.gemini = self.mock_gemini
    
    def test_init(self):
        """Test GeminiAgent initialization"""
        agent = GeminiAgent()
        
        # Check default values
        self.assertFalse(agent.verbose)
        self.assertTrue(isinstance(agent.tool_call_counts, dict))
        
        # Check tool call counts are initialized
        for key in ["segmentation", "depth", "novel_view", "point_cloud", "zoomed", "cropped", "counting"]:
            self.assertEqual(agent.tool_call_counts[key], 0)
    
    def test_get_prompt_capabilities(self):
        """Test the _get_prompt_capabilities method"""
        # Test with no capabilities
        agent = GeminiAgent(
            use_depth=False,
            use_segmentation=False,
            use_novel_view=False,
            use_point_cloud=False
        )
        prompt = agent._get_prompt_capabilities()
        self.assertEqual(prompt, "You are a helpful multimodal assistant.")
        
        # Test with some capabilities - need to properly mock modules availability
        agent = GeminiAgent(
            use_depth=False,  # Will be manually set after initialization
            use_segmentation=False,
            use_novel_view=False,
            use_point_cloud=False
        )
        
        # Manually set up the agent's capabilities
        agent.use_depth = True
        agent.use_segmentation = True
        
        # Mock the modules_available attribute to show modules are available
        agent.modules_available = {
            "depth": True,
            "segmentation": True,
            "novel_view": False,
            "point_cloud": False
        }
        
        # Now test the prompt
        prompt = agent._get_prompt_capabilities()
        self.assertIn("depth estimation", prompt)
        self.assertIn("segmentation", prompt)
        self.assertNotIn("novel view synthesis", prompt)
    
    def test_crop_and_zoom(self):
        """Test the crop_and_zoom method"""
        # Create a test image
        image = Image.new('RGB', (100, 100), color='red')
        
        # Crop the center 50x50 region
        bbox = (0.25, 0.25, 0.75, 0.75)  # normalized coordinates
        cropped = self.agent.crop_and_zoom(image, bbox)
        
        # Check dimensions
        self.assertEqual(cropped.size, (50, 50))
        
        # Check tool call count was incremented
        self.assertEqual(self.agent.tool_call_counts["cropped"], 1)
    
    @patch('coc.tree.gemi.Exec')
    def test_setup_exec_environment(self, mock_exec_class):
        """Test the setup_exec_environment method"""
        # Setup mock
        mock_exec = MockExec()
        mock_exec_class.return_value = mock_exec
        
        # Call the method
        result = self.agent.setup_exec_environment()
        
        # Check if Exec was called
        self.assertEqual(result, mock_exec)
        
        # Check if setup code was executed
        self.assertTrue(len(mock_exec.run_history) > 0)
    
    @patch('coc.tree.gemi.Exec')
    def test_analyze_with_code_execution(self, mock_exec_class):
        """Test the analyze_with_code_execution method"""
        # Setup mocks
        mock_exec = MockExec()
        mock_exec_class.return_value = mock_exec
        
        # Mock Gemini response for code generation
        self.mock_gemini.run_freestyle.side_effect = [
            """
            ```python
            # Analysis code
            print("Analyzing image...")
            results = {"count": 23}
            print(f"Found {results['count']} objects")
            ```
            """,
            "Test response"  # This will be the interpretation response
        ]
        
        # Call the method
        prompt = "How many bottles are there?"
        result = self.agent.analyze_with_code_execution(prompt, [self.test_image])
        
        # Check the result
        self.assertIn("Test response", result)
        
        # Check that Gemini was called twice (once for code generation, once for interpretation)
        self.assertEqual(self.mock_gemini.run_freestyle.call_count, 2)
        
        # Check that the images were added to the Exec environment
        self.assertTrue('image_0' in mock_exec.globals)
        self.assertTrue('images' in mock_exec.globals)
    
    def test_generate_with_no_images(self):
        """Test generate method with no images"""
        result = self.agent.generate("Test prompt", None)
        
        # Check that Gemini was called
        self.mock_gemini.run_freestyle.assert_called_once()
        
        # Check the result
        self.assertEqual(result, "Test response")
    
    def test_generate_orchestrated_counting_task(self):
        """Test generate_orchestrated method with a counting task"""
        # Mock analyze_with_code_execution
        self.agent.analyze_with_code_execution = MagicMock(return_value="Code execution result")
        
        # Call the method with a counting task
        prompt = "How many bottles are there?"
        result = self.agent.generate_orchestrated(prompt, [self.test_image])
        
        # Check that analyze_with_code_execution was called
        self.agent.analyze_with_code_execution.assert_called_once_with(prompt, [self.test_image])
        
        # Check the result
        self.assertEqual(result, "Code execution result")


class TestTreeNodeGeneration(unittest.TestCase):
    """Test the tree node generation functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a mock GeminiAgent
        self.mock_agent = MagicMock()
        self.mock_agent.generate.return_value = "def test_func():\n    print('Hello world')"
        self.mock_agent.analyze_with_code_execution.return_value = "def analyze():\n    print('Analysis')"
        
        # Create a test TreeNode with explicitly mocked create_child method
        self.parent_node = MagicMock()
        self.parent_node.curr_code = "def parent_func():\n    pass"
        
        # Add create_child method to the mock
        child_node = MagicMock()
        self.parent_node.create_child = MagicMock(return_value=child_node)
    
    def test_generate_one_child(self):
        """Test the generate_one_child function"""
        # Call the function
        child, error = gemi_generate_one_child(self.parent_node, "Improve the code", self.mock_agent)
        
        # Check that agent.generate was called
        self.mock_agent.generate.assert_called_once()
        
        # Check that a child was created
        self.parent_node.create_child.assert_called_once()
        
        # Check there was no error
        self.assertIsNone(error)
    
    def test_generate_one_child_with_images(self):
        """Test generate_one_child with images"""
        # Add images to the parent node
        self.parent_node.images = [Image.new('RGB', (10, 10))]
        
        # Call the function with a counting task
        child, error = gemi_generate_one_child(self.parent_node, "Improve counting", self.mock_agent)
        
        # Check that analyze_with_code_execution was called instead of generate
        self.mock_agent.analyze_with_code_execution.assert_called_once()
        
        # Check there was no error
        self.assertIsNone(error)
    
    def test_generate_children(self):
        """Test the generate_children function"""
        # Store original function references
        orig_generate_one_child = gemi_generate_one_child
        
        try:
            # Create a mock for generate_one_child function
            mock_child = MagicMock(spec=TreeNode)
            mock_generate_one_child = MagicMock(return_value=(mock_child, None))
            
            # Replace the imported function with our mock
            import coc.tree.gemi
            coc.tree.gemi.generate_one_child = mock_generate_one_child
            
            # Call generate_children
            nodes = [self.parent_node]
            children, errors = gemi_generate_children(nodes, 2, self.mock_agent)
            
            # Check that our mocked function was called twice
            self.assertEqual(mock_generate_one_child.call_count, 2)
            
            # Check we got the right children back
            self.assertEqual(len(children), 2)
            self.assertEqual(len(errors), 0)
            
        finally:
            # Restore original function
            coc.tree.gemi.generate_one_child = orig_generate_one_child


if __name__ == '__main__':
    unittest.main() 