import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os
import json
import asyncio
from PIL import Image
import io
import base64
from coc.tree.qua import (
    WebUIWrapper,
    generate_one_child,
    generate_children,
    generate_children_webui,
    evaluate,
    evaluate_with_webui,
    judge_if_any,
    run_with_webui,
    eval_a_batch
)
from coc.tree.une import TreeNode, CodeList, root_factory
from coc.tool.task import Task
from coc.exec import CONTEXT_FILE
from coc.data.fulltask import FullTask
from langchain_core.messages import AIMessage
from coc.util import Pair
from coc.util.text import extract_code, extract_boxed
from coc.prompts.prompt_troi import prompts_2nd_person
from coc.exec.context.prompt_brev import build_trunk
from coc.tool.vqa import gemini_as_llm
from coc.util.misc import fulltask_to_task, set_seed
from coc.tree.webui_config import (
    WEBUI_API_BASE_URL,
    DEFAULT_MODEL,
    MAX_CONCURRENT_REQUESTS,
    CONNECTION_TIMEOUT,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
)

class MockResponse:
    def __init__(self, status=200, data=None):
        self.status = status
        self.data = data or {"response": "test response"}
    
    async def json(self):
        return self.data
    
    async def text(self):
        return json.dumps(self.data)
    
    def __await__(self):
        async def _await_impl():
            return self
        return _await_impl().__await__()

class MockClientSession:
    def __init__(self, mock_responses=None):
        self.mock_responses = mock_responses or [MockResponse()]
        self.response_index = 0
        self.requests = []
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def post(self, url, json=None, **kwargs):
        self.requests.append((url, json, kwargs))
        resp = self.mock_responses[self.response_index]
        self.response_index = (self.response_index + 1) % len(self.mock_responses)
        return resp

class TestWebUIWrapper(unittest.TestCase):
    def setUp(self):
        # Create a test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        
        # Create patcher for aiohttp.ClientSession
        self.session_patcher = patch('aiohttp.ClientSession', return_value=MockClientSession())
        self.mock_session = self.session_patcher.start()
    
    def tearDown(self):
        self.session_patcher.stop()
    
    def test_encode_image(self):
        wrapper = WebUIWrapper()
        # Test with PIL Image
        encoded = wrapper._encode_image(self.test_image)
        self.assertTrue(encoded.startswith("data:image/png;base64,"))
        
        # Test with string path
        path_string = "/path/to/image.jpg"
        self.assertEqual(wrapper._encode_image(path_string), path_string)
        
        # Test with base64 string
        base64_string = "data:image/jpeg;base64,abc123"
        self.assertEqual(wrapper._encode_image(base64_string), base64_string)
    
    def test_run_freestyle(self):
        with patch.object(WebUIWrapper, 'generate', return_value="Test response") as mock_generate:
            wrapper = WebUIWrapper()
            
            # Test with empty inputs
            self.assertEqual(wrapper.run_freestyle([]), "")
            
            # Test with text only
            result = wrapper.run_freestyle(["Hello"])
            mock_generate.assert_called_with("Hello", None, None)
            
            # Test with text and images
            result = wrapper.run_freestyle(["Hello", self.test_image])
            mock_generate.assert_called_with("Hello", [self.test_image], None)
    
    @patch('asyncio.run')
    def test_generate(self, mock_run):
        mock_run.return_value = "Async response"
        wrapper = WebUIWrapper()
        
        result = wrapper.generate("Hello", images=[self.test_image], model="test-model")
        
        # Check that asyncio.run was called with generate_async
        mock_run.assert_called_once()
        self.assertEqual(result, "Async response")
    
    @patch('asyncio.run')
    def test_generate_branches(self, mock_run):
        mock_run.return_value = ["Response 1", "Response 2"]
        wrapper = WebUIWrapper()
        
        result = wrapper.generate_branches(
            ["Prompt 1", "Prompt 2"], 
            images=[self.test_image], 
            model="test-model"
        )
        
        # Check that asyncio.run was called with generate_branches_async
        mock_run.assert_called_once()
        self.assertEqual(result, ["Response 1", "Response 2"])

class TestTreeSearchFunctions(unittest.TestCase):
    def setUp(self):
        # Create mock task with images
        self.mock_task = {
            'query': 'Test query',
            'images': [Image.new('RGB', (100, 100), color='blue')],
            'choices': ['A', 'B', 'C', 'D'],
            'answer': 'A'
        }
        
        # Create mock codelist
        self.codelist = CodeList(CONTEXT_FILE, self.mock_task)
        
        # Create root node
        self.root = TreeNode(
            codelist=self.codelist,
            outputs=[],
            parent=None,
            children=[],
            depth=0
        )
    
    @patch('coc.tree.qua.extract_code')
    @patch('coc.tree.qua.extract_boxed')
    def test_generate_one_child_with_code(self, mock_extract_boxed, mock_extract_code):
        # Mock VLM to return code
        mock_vlm = MagicMock(return_value="Response with code")
        mock_extract_code.return_value = ["print('test')"]
        mock_extract_boxed.return_value = None
        
        child, answer = generate_one_child(self.root, "hint", mock_vlm)
        
        # Verify child node was created properly
        self.assertEqual(child.parent, self.root)
        self.assertEqual(child.depth, 1)
        self.assertEqual(len(child.codelist._), 1)
        self.assertIsNone(answer)
        
        # Verify VLM was called with correct arguments
        mock_vlm.assert_called_once()
    
    @patch('coc.tree.qua.extract_code')
    @patch('coc.tree.qua.extract_boxed')
    def test_generate_one_child_with_answer(self, mock_extract_boxed, mock_extract_code):
        # Mock VLM to return answer
        mock_vlm = MagicMock(return_value="Response with answer")
        mock_extract_code.return_value = []
        mock_extract_boxed.return_value = "A"
        
        child, answer = generate_one_child(self.root, "hint", mock_vlm)
        
        # Verify child node was created properly with answer
        self.assertEqual(child.parent, self.root)
        self.assertEqual(child.depth, 1)
        self.assertEqual(len(child.codelist._), 0)  # No code added
        self.assertEqual(answer, "A")
    
    @patch('coc.tree.qua.random.choice')
    @patch('coc.tree.qua.generate_one_child')
    def test_generate_children(self, mock_generate_one_child, mock_random_choice):
        # Setup mocks
        mock_random_choice.side_effect = lambda x: x[0]  # Always choose first item
        
        # Mock two different results: one with code, one with answer
        mock_generate_one_child.side_effect = [
            (TreeNode(self.codelist, ["output1"], self.root, [], 1), None),
            (TreeNode(self.codelist, ["output2"], self.root, [], 1), "A")
        ]
        
        children, answers = generate_children([self.root], 2, vlm=MagicMock())
        
        # Verify results
        self.assertEqual(len(children), 2)
        self.assertEqual(len(answers), 1)
        self.assertEqual(answers[0][1], "A")
    
    @patch('coc.tree.qua.gemini_as_llm')
    def test_judge_if_any(self, mock_gemini_as_llm):
        # Test when answer is correct
        mock_gemini_as_llm.return_value = "Yes, the answer matches. True"
        self.assertTrue(judge_if_any(["A", "B"], "A"))
        
        # Test when answer is incorrect
        mock_gemini_as_llm.return_value = "No matches found. False"
        self.assertFalse(judge_if_any(["B", "C"], "A"))
    
    @patch('coc.tree.qua.evaluate')
    @patch('coc.tree.qua.judge_if_any')
    @patch('coc.tree.qua.fulltask_to_task')
    def test_eval_a_batch(self, mock_fulltask_to_task, mock_judge_if_any, mock_evaluate):
        # Setup mocks
        mock_fulltask_to_task.return_value = self.mock_task
        mock_evaluate.return_value = [(MagicMock(), "A")]  # One node with answer "A"
    
        # First task gets correct answer, second task gets incorrect
        mock_judge_if_any.side_effect = [True, False]
    
        # Create mock fulltasks
        mock_fulltasks = [
            {'query': 'Test 1', 'answer': 'A', 'images': []},
            {'query': 'Test 2', 'answer': 'B', 'images': []}
        ]
    
        # Test with first task correct
        correct, total = eval_a_batch(mock_fulltasks, MagicMock())
        self.assertEqual(correct, 1)
        self.assertEqual(total, 2)

class TestWebUIIntegration(unittest.TestCase):
    def setUp(self):
        # Create mock WebUIWrapper
        self.mock_webui = MagicMock(spec=WebUIWrapper)
        
        # Create mock task
        self.mock_task = {
            'query': 'Test query',
            'images': [Image.new('RGB', (100, 100), color='blue')],
            'choices': ['A', 'B', 'C', 'D'],
            'answer': 'A'
        }
        
        # Create mock fulltask
        self.mock_fulltask = FullTask({
            'query': 'Test query',
            'choices': ['A', 'B', 'C', 'D'],
            'answer': 'A',
            'images': []
        })
        
        # Create mock codelist and root node
        self.codelist = CodeList(CONTEXT_FILE, self.mock_task)
        self.root = TreeNode(
            codelist=self.codelist,
            outputs=[],
            parent=None,
            children=[],
            depth=0
        )
    
    @patch('coc.tree.qua.root_factory')
    @patch('coc.tree.qua.generate_children_webui')
    def test_evaluate_with_webui(self, mock_generate_children_webui, mock_root_factory):
        # Setup mocks
        mock_root_factory.return_value = self.root
        
        # Mock generate_children_webui to return nodes with answers on 2nd call
        child_node = TreeNode(self.codelist, ["output"], self.root, [], 1)
        mock_generate_children_webui.side_effect = [
            ([child_node], []),  # First call returns a node with no answer
            ([], [(child_node, "A")])  # Second call returns node with answer
        ]
        
        # Test evaluate_with_webui
        result = evaluate_with_webui(self.mock_task, self.mock_webui)
        
        # Verify results
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], "A")
    
    @patch('coc.tree.qua.WebUIWrapper')
    @patch('coc.tree.qua.evaluate_with_webui')
    @patch('coc.tree.qua.judge_if_any')
    @patch('coc.tree.qua.fulltask_to_task')
    def test_run_with_webui(self, mock_fulltask_to_task, mock_judge_if_any, 
                           mock_evaluate_with_webui, mock_webui_wrapper):
        # Setup mocks
        mock_webui_instance = MagicMock()
        mock_webui_wrapper.return_value = mock_webui_instance
        mock_fulltask_to_task.return_value = self.mock_task
        
        # Mock evaluate_with_webui to return answers
        child_node = TreeNode(self.codelist, ["output"], self.root, [], 1)
        mock_evaluate_with_webui.return_value = [(child_node, "A")]
        
        # Mock judge_if_any to return True
        mock_judge_if_any.return_value = True
        
        # Test run_with_webui
        correct, total = run_with_webui([self.mock_fulltask], model_name="test-model")
        
        # Verify results
        self.assertEqual(correct, 1)
        self.assertEqual(total, 1)
        
        # Verify WebUIWrapper was created with correct parameters
        mock_webui_wrapper.assert_called_once()
        mock_webui_instance.default_model = "test-model"

if __name__ == '__main__':
    unittest.main() 