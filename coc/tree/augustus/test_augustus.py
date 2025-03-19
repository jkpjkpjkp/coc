import unittest
from typing import List, Optional, Tuple, Dict, Any, Union
from PIL import Image
import numpy as np
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Import the components we need to test
from coc.tool.task import Task
from coc.tool.vqa.gemini import Gemini
from coc.exec.mod import Exec
from coc.tree.augustus.augustus import AugustusNode, AugustusTree, execute_react_step


class TestAugustusSanity(unittest.TestCase):
    
    def setUp(self):
        # Create a mock task with dummy image
        sample_image = Image.new('RGB', (100, 100), color='red')
        self.mock_task = {'images': [sample_image], 'question': 'What is the color of this image?'}
        
        # Create a mock Gemini instance
        self.mock_gemini = MagicMock(spec=Gemini)
        self.mock_gemini.run_freestyle.return_value = "I'll write code to analyze the image:\n```python\nimport numpy as np\nfrom PIL import Image\n\n# Get the image\nimg = task['images'][0]\n\n# Analyze dominant color\npixels = np.array(img)\nmean_color = pixels.mean(axis=(0,1))\nprint(f\"Mean RGB: {mean_color}\")\n```\n\nThe mean RGB values show the image is primarily red."
    
    def test_execute_react_step(self):
        # Test the execute_react_step function
        exec_env = Exec()
        exec_env.set_var('task', self.mock_task)
        
        # Execute a step with the mock Gemini
        code, response, is_final = execute_react_step(self.mock_gemini, exec_env, "Analyze the image color")
        
        # Verify we got code in the response
        self.assertIsNotNone(code)
        self.assertIn("import numpy", code)
        
        # Verify the response contains analysis
        self.assertIn("red", response)
        
        # This example shouldn't be final yet
        self.assertFalse(is_final)
    
    def test_augustus_node(self):
        # Test node creation and state management
        exec_env = Exec()
        exec_env.set_var('task', self.mock_task)
        
        # Create a node
        node = AugustusNode(
            exec_env=exec_env,
            prompt="Analyze the image color",
            parent=None,
            depth=0
        )
        
        # Check that the node has the correct properties
        self.assertEqual(node.depth, 0)
        self.assertIsNone(node.parent)
        self.assertEqual(node.prompt, "Analyze the image color")
        self.assertEqual(len(node.children), 0)
    
    def test_augustus_tree_search(self):
        # Test the full tree search with a mock Gemini
        with patch('coc.tree.augustus.augustus.Gemini') as mock_gemini_class:
            mock_gemini_class.return_value = self.mock_gemini
            
            # Create a tree
            tree = AugustusTree(max_depth=2, branch_factor=2)
            
            # Run the search
            result = tree.search(self.mock_task)
            
            # Verify we get a result
            self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main() 