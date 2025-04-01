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
    GeminiOpenAIWrapper,
    create_vlm_wrapper,
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
from coc.tool.context.prompt_brev import build_trunk
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
    GEMINI_BASE_URL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    USE_OPENAI_FORMAT,
)
import numpy as np

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

class TestGeminiOpenAIWrapper(unittest.TestCase):
    def setUp(self):
        # Create a test image
        self.test_image = Image.new('RGB', (100, 100), color='red')

        # Mock async response
        self.mock_response = MagicMock()
        self.mock_response.status = 200
        self.mock_response.json = AsyncMock(return_value={
            "choices": [
                {
                    "message": {
                        "content": "Test response from Gemini",
                        "role": "assistant"
                    },
                    "index": 0
                }
            ]
        })

        # Create patcher for aiohttp.ClientSession
        self.mock_session = AsyncMock()
        self.mock_session.__aenter__ = AsyncMock(return_value=self.mock_session)
        self.mock_session.__aexit__ = AsyncMock(return_value=None)
        self.mock_session.post = AsyncMock(return_value=self.mock_response)

        # Patch ClientSession
        self.session_patcher = patch('aiohttp.ClientSession', return_value=self.mock_session)
        self.mock_client_session = self.session_patcher.start()

    def tearDown(self):
        self.session_patcher.stop()

    def test_encode_image(self):
        wrapper = GeminiOpenAIWrapper()
        # Test with PIL Image
        encoded = wrapper._encode_image(self.test_image)
        self.assertTrue(isinstance(encoded, str))
        self.assertTrue(len(encoded) > 0)

        # Test with string path
        path_string = "/path/to/image.jpg"
        self.assertEqual(wrapper._encode_image(path_string), path_string)

    def test_prepare_messages(self):
        wrapper = GeminiOpenAIWrapper()

        # Test text-only message
        messages = wrapper._prepare_messages("Hello")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "Hello")

        # Test with image
        messages = wrapper._prepare_messages("Describe this image", [self.test_image])
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1]["role"], "user")
        self.assertIsInstance(messages[1]["content"], list)
        self.assertEqual(len(messages[1]["content"]), 2)  # Text + 1 image
        self.assertEqual(messages[1]["content"][0]["type"], "text")
        self.assertEqual(messages[1]["content"][1]["type"], "image_url")

    @patch('asyncio.run')
    def test_generate(self, mock_run):
        mock_run.return_value = "Async response"
        wrapper = GeminiOpenAIWrapper()

        result = wrapper.generate("Hello", images=None, model="gemini-pro")

        # Check that asyncio.run was called with generate_async
        mock_run.assert_called_once()
        self.assertEqual(result, "Async response")

    @patch('asyncio.run')
    def test_generate_branches(self, mock_run):
        mock_run.return_value = ["Response 1", "Response 2"]
        wrapper = GeminiOpenAIWrapper()

        result = wrapper.generate_branches(
            ["Prompt 1", "Prompt 2"],
            images=None,
            model="gemini-pro"
        )

        # Check that asyncio.run was called with generate_branches_async
        mock_run.assert_called_once()
        self.assertEqual(result, ["Response 1", "Response 2"])

    @patch('asyncio.gather')
    @patch('coc.tree.qua.GeminiOpenAIWrapper.generate_async')
    async def test_generate_branches_async(self, mock_generate_async, mock_gather):
        wrapper = GeminiOpenAIWrapper()

        mock_generate_async.return_value = "Response"
        mock_gather.return_value = ["Response 1", "Response 2"]

        result = await wrapper.generate_branches_async(
            ["Prompt 1", "Prompt 2"],
            images=None,
            model="gemini-pro"
        )

        self.assertEqual(mock_generate_async.call_count, 2)
        self.assertEqual(result, ["Response 1", "Response 2"])

class TestWrapperFactory(unittest.TestCase):
    def test_create_vlm_wrapper(self):
        # Test creating GeminiOpenAIWrapper
        wrapper = create_vlm_wrapper(use_gemini=True, base_url="http://gemini-broker.com")
        self.assertIsInstance(wrapper, GeminiOpenAIWrapper)
        self.assertEqual(wrapper.base_url, "http://gemini-broker.com")

# Test integrating Gemini wrapper with the evaluation functions
class TestGeminiIntegration(unittest.TestCase):
    def setUp(self):
        # Create mock task with images
        self.mock_task = {
            'query': 'Test query',
            'images': [Image.new('RGB', (100, 100), color='blue')],
            'choices': ['A', 'B', 'C', 'D'],
            'answer': 'A'
        }

        # Mock wrapper
        self.mock_wrapper = MagicMock()
        self.mock_wrapper.run_freestyle.return_value = "```python\nprint('hello')\n```"
        self.mock_wrapper.generate_branches.return_value = [
            "```python\nprint('test 1')\n```",
            "```python\nprint('test 2')\n```"
        ]

    @patch('coc.tree.qua.GeminiOpenAIWrapper')
    @patch('coc.tree.qua.evaluate_with_webui')
    @patch('coc.tree.qua.judge_if_any')
    @patch('coc.tree.qua.fulltask_to_task')
    def test_run_with_webui_using_gemini(self, mock_fulltask_to_task, mock_judge_if_any,
                                         mock_evaluate_with_webui, mock_gemini_wrapper):
        # Configure mocks
        mock_fulltask_to_task.return_value = self.mock_task
        mock_judge_if_any.return_value = True
        mock_evaluate_with_webui.return_value = [(None, "A")]
        mock_gemini_instance = mock_gemini_wrapper.return_value

        # Mock full tasks
        mock_fulltasks = [{'query': 'Test 1', 'answer': 'A'}]

        # Run the function with Gemini wrapper
        correct, total = run_with_webui(
            mock_fulltasks,
            model_name="gemini-pro-vision",
            base_url="http://gemini-broker.com",
            api_key="test-key",
            use_gemini=True
        )

        # Verify we used the Gemini wrapper
        mock_gemini_wrapper.assert_called_once_with(
            base_url="http://gemini-broker.com",
            api_key="test-key"
        )

        # Check model was set
        self.assertEqual(mock_gemini_instance.model, "gemini-pro-vision")

        # Verify correct results
        self.assertEqual(correct, 1)
        self.assertEqual(total, 1)

class TestImageVariables(unittest.TestCase):
    
    def setUp(self):
        # Create mock images
        self.mock_image1 = Image.new('RGB', (10, 10), color='red')
        self.mock_image2 = Image.new('RGB', (10, 10), color='blue')
        
        # Create a task with images
        self.task = Task(
            images=[self.mock_image1, self.mock_image2],
            question="Test question"
        )
        
        # Create a mock VLM function
        self.mock_vlm = MagicMock()
        self.mock_vlm.return_value = "```python\nprint('test')\n```"
    
    def test_image_variables_in_exec_env(self):
        """Test that image variables are correctly set in the exec environment"""
        root = root_factory(self.task)
        
        # Check that image variables exist in the environment
        self.assertIn('image_1', root.codelist.env.globals)
        self.assertIn('image_2', root.codelist.env.globals)
        
        # Check that the images are correct
        self.assertEqual(root.codelist.env.globals['image_1'], self.mock_image1)
        self.assertEqual(root.codelist.env.globals['image_2'], self.mock_image2)
    
    @patch('coc.tree.qua.build_trunk')
    def test_generate_one_child_uses_image_variables(self, mock_build_trunk):
        """Test that generate_one_child correctly uses image variables"""
        mock_build_trunk.return_value = "Test message"
        
        root = root_factory(self.task)
        
        # Call generate_one_child
        generate_one_child(root, "Test hint", self.mock_vlm)
        
        # Check that VLM was called with the correct arguments
        self.mock_vlm.assert_called_once()
        call_args = self.mock_vlm.call_args[0][0]
        
        # First argument should be the message
        self.assertEqual(call_args[0], "Test message")
        
        # Next arguments should be the images
        self.assertEqual(call_args[1], self.mock_image1)
        self.assertEqual(call_args[2], self.mock_image2)
        
    def test_deepcopy_preserves_image_variables(self):
        """Test that deepcopy preserves image variables"""
        root = root_factory(self.task)
        
        # Deepcopy the codelist
        copied_codelist = root.codelist.deepcopy()
        
        # Check that image variables exist in the copied environment
        self.assertIn('image_1', copied_codelist.env.globals)
        self.assertIn('image_2', copied_codelist.env.globals)
        
        # Check that the images are correct
        self.assertEqual(copied_codelist.env.globals['image_1'], self.mock_image1)
        self.assertEqual(copied_codelist.env.globals['image_2'], self.mock_image2)

if __name__ == '__main__':
    unittest.main()