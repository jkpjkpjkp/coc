import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from coc.config import MAX_DEPTH, BRANCH_WIDTH
from coc.tree.une import CodeList, TreeNode, root_factory
from coc.tool.task import Task
from typing import Optional, List, Iterable, Union
from coc.util.text import extract_code, extract_boxed
from coc.prompts.prompt_troi import prompts_2nd_person
from coc.tool.context.prompt_brev import build_trunk
from coc.tool.vqa import gemini_as_llm  # Used for text-only judgment
from coc.util.misc import fulltask_to_task, set_seed
from coc.data.fulltask import FullTask
import textwrap
import json
import os
import aiohttp
import asyncio
from PIL import Image
import base64
import io
import time
import logging
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

# Set up logging
logger = logging.getLogger(__name__)

# Note: Replace with actual import, e.g., 'from coc.tree.vlm import vlm'
# vlm is assumed to be a function accepting a list of Union[str, PIL.Image.Image] and returning a str
from coc.tool.vqa.gemini import Gemini
gemini = Gemini()
def vlm(*args, **kwargs):
    return gemini.run_freestyle(*args, **kwargs)

class GeminiOpenAIWrapper:
    """
    A wrapper that calls Gemini 2 Pro using the OpenAI API format via a broker.
    """
    def __init__(self, base_url=GEMINI_BASE_URL, api_key=GEMINI_API_KEY, model=GEMINI_MODEL):
        self.base_url = base_url
        self.api_url = f"{base_url}/v1/chat/completions"
        self.model = model
        self.api_key = api_key
        self.connection_timeout = CONNECTION_TIMEOUT
        self.request_timeout = REQUEST_TIMEOUT
        self.max_retries = MAX_RETRIES
        self.retry_delay = RETRY_DELAY
    
    def _get_headers(self):
        """Get headers for the API request, including API key"""
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _encode_image(self, image):
        """Convert PIL Image to base64 string for API consumption"""
        if isinstance(image, str):
            # If it's already a string path or base64, return as is
            return image
        
        # If it's a PIL Image, convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def _prepare_messages(self, prompt, images=None):
        """
        Prepare messages in the OpenAI chat format with images in content
        
        Args:
            prompt: String prompt to send
            images: List of PIL Images or image paths
            
        Returns:
            List of message objects in OpenAI format
        """
        # Start with the system message
        messages = [
            {"role": "system", "content": "You are a helpful multimodal assistant."}
        ]
        
        # Create user message with text and images
        if images:
            # Format with multiple image support
            content = [{"type": "text", "text": prompt}]
            
            for img in images:
                encoded_img = self._encode_image(img)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_img}"
                    }
                })
            
            messages.append({"role": "user", "content": content})
        else:
            # Text-only message
            messages.append({"role": "user", "content": prompt})
        
        return messages
    
    async def generate_async(self, prompt, images=None, model=None):
        """
        Generate a response using the Gemini API via OpenAI format.
        
        Args:
            prompt: String prompt to send to the model
            images: List of PIL Images or image paths
            model: Model to use (defaults to GEMINI_MODEL env var)
            
        Returns:
            String response from the model
        """
        messages = self._prepare_messages(prompt, images)
        
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False
        }
        
        for retry in range(self.max_retries):
            try:
                timeout = aiohttp.ClientTimeout(
                    connect=self.connection_timeout,
                    total=self.request_timeout
                )
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.api_url, 
                        json=payload,
                        headers=self._get_headers()
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            error_msg = f"API request failed with status {response.status}: {error_text}"
                            logger.error(error_msg)
                            if retry < self.max_retries - 1:
                                logger.info(f"Retrying in {self.retry_delay} seconds (attempt {retry+1}/{self.max_retries})")
                                await asyncio.sleep(self.retry_delay)
                                continue
                            else:
                                raise Exception(error_msg)
                        
                        # Parse the response
                        result = await response.json()
                        # Extract the content from the OpenAI format
                        if "choices" in result and len(result["choices"]) > 0:
                            message = result["choices"][0]["message"]
                            content = message.get("content", "")
                            return content
                        return ""
                        
            except asyncio.TimeoutError:
                if retry < self.max_retries - 1:
                    logger.warning(f"Request timed out. Retrying in {self.retry_delay} seconds (attempt {retry+1}/{self.max_retries})")
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error("All retry attempts failed due to timeout")
                    raise
            except Exception as e:
                if retry < self.max_retries - 1:
                    logger.warning(f"Error: {str(e)}. Retrying in {self.retry_delay} seconds (attempt {retry+1}/{self.max_retries})")
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"All retry attempts failed: {str(e)}")
                    raise
    
    def generate(self, prompt, images=None, model=None):
        """Synchronous wrapper around generate_async"""
        return asyncio.run(self.generate_async(prompt, images, model))
    
    def run_freestyle(self, inputs, model=None):
        """
        Compatible interface with gemini.run_freestyle.
        
        Args:
            inputs: List of inputs, with the first being the prompt and the rest being images
            
        Returns:
            String response from the model
        """
        if not inputs:
            return ""
        
        prompt = inputs[0] if inputs else ""
        images = inputs[1:] if len(inputs) > 1 else None
        
        return self.generate(prompt, images, model)
    
    async def generate_branches_async(self, prompts, images=None, model=None):
        """
        Generate multiple responses in parallel using the Gemini API.
        
        Args:
            prompts: List of string prompts to send to the model
            images: List of PIL Images or image paths (shared across all prompts)
            model: Model to use (defaults to GEMINI_MODEL env var)
            
        Returns:
            List of string responses from the model
        """
        if not prompts:
            return []
            
        tasks = []
        for prompt in prompts:
            tasks.append(self.generate_async(prompt, images, model))
            
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def generate_branches(self, prompts, images=None, model=None):
        """Synchronous wrapper around generate_branches_async"""
        return asyncio.run(self.generate_branches_async(prompts, images, model))

def generate_one_child(parent: TreeNode, suggestive_hint: str, vlm) -> tuple[TreeNode, Optional[str]]:
    """
    Generate a single child node from a parent using a suggestive prompt and images.

    Args:
        parent: Parent node to expand from.
        suggestive_hint: Prompt to guide the VLM's response.
        vlm: Vision-Language Model function accepting a list of Union[str, Image.Image].

    Returns:
        Tuple of (child node, answer if found, else None).
    """
    task = parent.codelist.env.get_var('task')
    message = build_trunk(
        task=task,
        codes=parent.codelist.to_list_of_pair_of_str(),
        hint=suggestive_hint,
    )
    # Pass text message followed by task images to the VLM
    response = vlm([message] + task['images'])
    codes = extract_code(response)
    answer = extract_boxed(response)

    # Ensure response contains either code or answer, but not both
    assert not (codes and answer), f"Both code and answer extracted: {response}"
    assert codes or answer, f"Neither code nor answer extracted: {response}"

    # Create a new child node
    child_codelist = parent.codelist.deepcopy()
    if answer:
        child = TreeNode(
            codelist=child_codelist,
            outputs=parent.outputs + [response],
            parent=parent,
            children=[],
            depth=parent.depth + 1
        )
        return child, answer
    else:
        for code in codes:
            child_codelist.append(code)
        child = TreeNode(
            codelist=child_codelist,
            outputs=parent.outputs + [response],
            parent=parent,
            children=[],
            depth=parent.depth + 1
        )
        return child, None

def generate_children(nodes_with_code: list[TreeNode], num_children: int, vlm) -> tuple[list[TreeNode], list[tuple[TreeNode, str]]]:
    """
    Generate multiple child nodes in parallel from the given parent nodes.

    Args:
        nodes_with_code: List of parent nodes that have code (not answers).
        num_children: Number of children to generate (BRANCH_WIDTH).
        vlm: Vision-Language Model function.

    Returns:
        Tuple of (list of child nodes, list of (node, answer) pairs).
    """
    with ThreadPoolExecutor(max_workers=num_children) as executor:
        futures = []
        for _ in range(num_children):
            parent = random.choice(nodes_with_code)
            suggestive_hint = random.choice(prompts_2nd_person)
            futures.append(executor.submit(generate_one_child, parent, suggestive_hint, vlm))

        children = []
        answers = []
        for future in as_completed(futures):
            child, answer = future.result()
            children.append(child)
            if answer:
                answers.append((child, answer))
        return children, answers

def evaluate(task: Task, vlm) -> list[tuple[TreeNode, str]]:
    """
    Evaluate the task by building a branching tree with suggestive prompts and images.

    Each depth has BRANCH_WIDTH nodes, generated by randomly selecting parents from the previous depth
    and using random suggestive prompts as VLM input prefixes, along with task images. Expands up to MAX_DEPTH.

    Args:
        task: The programming task to solve, including 'images' field.
        vlm: Vision-Language Model function.

    Returns:
        List of (node, answer) pairs found during tree construction.
    """
    root = root_factory(task)
    current_depth_nodes = [root]
    all_answers = []

    for depth in range(1, MAX_DEPTH + 1):
        nodes_with_code = [
            node for node in current_depth_nodes
            if not node.outputs or not extract_boxed(node.outputs[-1])
        ]
        if not nodes_with_code:
            break
        children, answers = generate_children(nodes_with_code, BRANCH_WIDTH, vlm)
        all_answers.extend(answers)
        current_depth_nodes = children

    return all_answers

def judge_if_any(outputs: List[str], answer: str) -> bool:
    """
    Judge if any output is correct using a text-only LLM.

    Args:
        outputs: List of output strings from the VLM.
        answer: Correct answer string.

    Returns:
        True if any output matches the answer, False otherwise.
    """
    ret = gemini_as_llm(textwrap.dedent(f'''
        judge the following outputs and end your return with True if ANY one of them is correct,
            otherwise end with False.
        outputs: {outputs}
        answer (correct choice is): {answer}'''
    ))
    return 'False' not in ret and 'True' in ret

def eval_a_batch(batch: Iterable[FullTask], vlm) -> tuple[int, int]:
    """Evaluate a batch of tasks.
    
    Args:
        batch: List of FullTask objects to evaluate
        vlm: Visual language model to use for evaluation
        
    Returns:
        Tuple of (number of correct answers, total number of tasks)
    """
    correct = 0
    total = 0
    
    for fulltask in batch:
        task = fulltask_to_task(fulltask)
        nodes = evaluate(task, vlm)
        
        # Get all answers for this task
        answers = [answer for _, answer in nodes]
        
        # Check if any answer is correct
        if judge_if_any(answers, task['answer']):
            correct += 1
        total += 1
    
    return correct, total

def create_vlm_wrapper(use_gemini=USE_OPENAI_FORMAT, **kwargs):
    """
    Factory function to create the appropriate VLM wrapper based on configuration.
    
    Args:
        use_gemini: Whether to use the Gemini OpenAI format wrapper
        **kwargs: Additional arguments to pass to the wrapper constructor
        
    Returns:
        An instance of WebUIWrapper or GeminiOpenAIWrapper
    """
    if use_gemini:
        return GeminiOpenAIWrapper(**kwargs)
    else:
        raise NotImplementedError("WebUI is not supported")

if __name__ == '__main__':
    print('A')
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate programming tasks using tree search with VLM')
    parser.add_argument('--model', type=str, choices=['gemini', 'webui', 'gemini-openai'], default='gemini',
                        help='VLM backend to use (default: gemini)')
    parser.add_argument('--webui-api-key', type=str, default=None,
                        help='API key for WebUI authentication (default: read from WEBUI_API_KEY env var)')
    parser.add_argument('--gemini-url', type=str, default=GEMINI_BASE_URL,
                        help=f'Base URL for the Gemini OpenAI format broker (default: {GEMINI_BASE_URL})')
    parser.add_argument('--gemini-model', type=str, default=GEMINI_MODEL,
                        help=f'Model name to use with Gemini (default: {GEMINI_MODEL})')
    parser.add_argument('--gemini-api-key', type=str, default=None,
                        help='API key for Gemini authentication (default: read from GEMINI_API_KEY env var)')
    parser.add_argument('--offer', type=str, default='sub',
                        help='Task offering to evaluate (default: sub)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None)')
    
    args = parser.parse_args()
    
    # Set seed if provided
    if args.seed is not None:
        set_seed(args.seed)
    else:
        set_seed()
    # Load tasks
    from coc.data.zero import zero
    print('C')
    batch = list(zero(offer=args.offer))
    print('D')
    # Run evaluation with selected model
    if args.model == 'gemini':
        print(f"Evaluating {len(batch)} tasks using Gemini...")
        correct, total = eval_a_batch(batch, vlm)
    else:
        raise NotImplementedError(f"Model {args.model} not supported")
    
    print(f"Correct: {correct}, Total: {total}, Accuracy: {correct/total:.2%}")