import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from coc.config import MAX_DEPTH, BRANCH_WIDTH
from coc.tree.une import CodeList, TreeNode, root_factory
from coc.tool.task import Task
from typing import Optional, List, Iterable, Union
from coc.util.text import extract_code, extract_boxed
from coc.prompts.prompt_troi import prompts_2nd_person
from coc.exec.context.prompt_brev import build_trunk
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
)

# Set up logging
logger = logging.getLogger(__name__)

# Note: Replace with actual import, e.g., 'from coc.tree.vlm import vlm'
# vlm is assumed to be a function accepting a list of Union[str, PIL.Image.Image] and returning a str
from coc.tool.vqa.gemini import Gemini
gemini = Gemini()
def vlm(*args, **kwargs):
    return gemini.run_freestyle(*args, **kwargs)

class WebUIWrapper:
    """
    A wrapper around the open-webui API for VLM functionality.
    """
    def __init__(self, base_url=WEBUI_API_BASE_URL, api_key=None):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.default_model = DEFAULT_MODEL
        self.max_concurrent = MAX_CONCURRENT_REQUESTS
        self.connection_timeout = CONNECTION_TIMEOUT
        self.request_timeout = REQUEST_TIMEOUT
        self.max_retries = MAX_RETRIES
        self.retry_delay = RETRY_DELAY
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.environ.get("WEBUI_API_KEY", "")
    
    def _encode_image(self, image):
        """Convert PIL Image to base64 string for API consumption"""
        if isinstance(image, str):
            # If it's already a string path or base64, return as is
            return image
        
        # If it's a PIL Image, convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
    
    def _get_headers(self):
        """Get headers for the API request, including API key if available"""
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def generate_async(self, prompt, images=None, model=None):
        """
        Generate a response using the open-webui API.
        
        Args:
            prompt: String prompt to send to the model
            images: List of PIL Images or image paths
            model: Model to use (defaults to DEFAULT_MODEL env var or llava:latest)
            
        Returns:
            String response from the model
        """
        encoded_images = []
        if images:
            encoded_images = [self._encode_image(img) for img in images]
        
        payload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "images": encoded_images if encoded_images else None,
            "stream": False  # We want the complete response at once
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
                        return result.get("response", "")
                        
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
        Generate multiple responses in parallel using the API.
        
        Args:
            prompts: List of string prompts to send to the model
            images: List of PIL Images or image paths (shared across all prompts)
            model: Model to use (defaults to DEFAULT_MODEL env var or llava:latest)
            
        Returns:
            List of string responses from the model
        """
        if not prompts:
            return []
            
        encoded_images = []
        if images:
            encoded_images = [self._encode_image(img) for img in images]
        
        # Prepare all payloads
        payloads = []
        for prompt in prompts:
            payload = {
                "model": model or self.default_model,
                "prompt": prompt,
                "images": encoded_images if encoded_images else None,
                "stream": False
            }
            payloads.append(payload)
        
        # Get headers with API key
        headers = self._get_headers()
        
        # Process in batches to avoid overwhelming the server
        responses = []
        for i in range(0, len(payloads), self.max_concurrent):
            batch = payloads[i:i + self.max_concurrent]
            
            # Create a timeout for connections and requests
            timeout = aiohttp.ClientTimeout(
                connect=self.connection_timeout,
                total=self.request_timeout
            )
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                tasks = []
                for payload in batch:
                    for retry in range(self.max_retries):
                        try:
                            task = session.post(self.api_url, json=payload, headers=headers)
                            tasks.append(task)
                            break
                        except Exception as e:
                            if retry < self.max_retries - 1:
                                logger.warning(f"Connection error: {str(e)}. Retrying in {self.retry_delay}s")
                                await asyncio.sleep(self.retry_delay)
                            else:
                                logger.error(f"Failed to create request after {self.max_retries} retries: {str(e)}")
                                responses.append(f"Failed to create request: {str(e)}")
                
                batch_responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process responses
                for resp in batch_responses:
                    if isinstance(resp, Exception):
                        responses.append(f"Request failed with exception: {str(resp)}")
                    elif isinstance(resp, aiohttp.ClientResponse):
                        if resp.status != 200:
                            error_text = await resp.text()
                            responses.append(f"API request failed with status {resp.status}: {error_text}")
                        else:
                            try:
                                result = await resp.json()
                                responses.append(result.get("response", ""))
                            except Exception as e:
                                responses.append(f"Failed to parse response: {str(e)}")
                    else:
                        responses.append(f"Unexpected response type: {type(resp)}")
        
        return responses
    
    def generate_branches(self, prompts, images=None, model=None):
        """Synchronous wrapper around generate_branches_async"""
        return asyncio.run(self.generate_branches_async(prompts, images, model))

# Example usage of WebUIWrapper:
# webui = WebUIWrapper()
# def webui_vlm(*args, **kwargs):
#     return webui.run_freestyle(*args, **kwargs)

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

def run_with_webui(batch: Iterable[FullTask], model_name=None, base_url="http://localhost:8080", api_key=None) -> tuple[int, int]:
    """
    Evaluate a batch of tasks using the WebUI interface for multimodal interaction.
    
    Args:
        batch: Iterable of FullTask objects, each potentially including images.
        model_name: Model to use for generation (e.g., "llava:latest")
        base_url: Base URL for the open-webui API
        api_key: API key for authentication (or None to use environment variables)
        
    Returns:
        Tuple of (number of correct answers, total tasks).
    """
    webui = WebUIWrapper(base_url=base_url, api_key=api_key)
    if model_name:
        webui.default_model = model_name
    
    correct = 0
    total = 0
    
    for fulltask in batch:
        task = fulltask_to_task(fulltask)
        answers_nodes = evaluate_with_webui(task, webui)
        
        if answers_nodes:
            answers = [answer for _, answer in answers_nodes]
            most_common_answer = max(set(answers), key=answers.count)
            if judge_if_any(answers, task['answer']):
                correct += 1
        total += 1
    
    return correct, total

# Enhanced function to use WebUIWrapper's branching capability
def generate_children_webui(nodes_with_code: list[TreeNode], num_children: int, webui: WebUIWrapper) -> tuple[list[TreeNode], list[tuple[TreeNode, str]]]:
    """
    Generate multiple child nodes in parallel using WebUI branching capability.
    
    Args:
        nodes_with_code: List of parent nodes that have code (not answers).
        num_children: Number of children to generate (BRANCH_WIDTH).
        webui: WebUIWrapper instance.
        
    Returns:
        Tuple of (list of child nodes, list of (node, answer) pairs).
    """
    # Create parent-hint pairs
    parent_hint_pairs = []
    for _ in range(num_children):
        parent = random.choice(nodes_with_code)
        suggestive_hint = random.choice(prompts_2nd_person)
        parent_hint_pairs.append((parent, suggestive_hint))
    
    # Build prompts for each parent-hint pair
    prompts = []
    for parent, hint in parent_hint_pairs:
        task = parent.codelist.env.get_var('task')
        prompt = build_trunk(
            task=task,
            codes=parent.codelist.to_list_of_pair_of_str(),
            hint=hint
        )
        prompts.append(prompt)
    
    # Get task images from the first parent (they should all have the same task)
    task = parent_hint_pairs[0][0].codelist.env.get_var('task')
    task_images = task['images']
    
    # Generate responses in parallel
    responses = webui.generate_branches(prompts, images=task_images)
    
    # Process responses
    children = []
    answers = []
    
    for i, response in enumerate(responses):
        parent, _ = parent_hint_pairs[i]
        codes = extract_code(response)
        answer = extract_boxed(response)
        
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
            children.append(child)
            answers.append((child, answer))
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
            children.append(child)
    
    return children, answers

def evaluate_with_webui(task: Task, webui: WebUIWrapper) -> list[tuple[TreeNode, str]]:
    """
    Evaluate the task by building a branching tree with WebUI branching capability.
    
    Args:
        task: The programming task to solve, including 'images' field.
        webui: WebUIWrapper instance.
        
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
        
        # Use the WebUI branching capability
        children, answers = generate_children_webui(nodes_with_code, BRANCH_WIDTH, webui)
        all_answers.extend(answers)
        current_depth_nodes = children
    
    return all_answers

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate programming tasks using tree search with VLM')
    parser.add_argument('--model', type=str, choices=['gemini', 'webui'], default='webui',
                        help='VLM backend to use (default: webui)')
    parser.add_argument('--webui-url', type=str, default=WEBUI_API_BASE_URL,
                        help=f'Base URL for the WebUI API (default: {WEBUI_API_BASE_URL})')
    parser.add_argument('--webui-model', type=str, default=DEFAULT_MODEL,
                        help=f'Model name to use with WebUI (default: {DEFAULT_MODEL})')
    parser.add_argument('--webui-api-key', type=str, default=None,
                        help='API key for WebUI authentication (default: read from WEBUI_API_KEY env var)')
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
    batch = zero(offer=args.offer)
    
    # Run evaluation with selected model
    if args.model == 'gemini':
        print(f"Evaluating {len(batch)} tasks using Gemini...")
        correct, total = eval_a_batch(batch, vlm)
    else:
        print(f"Evaluating {len(batch)} tasks using WebUI ({args.webui_model})...")
        print(f"WebUI URL: {args.webui_url}")
        api_key = args.webui_api_key or os.environ.get("WEBUI_API_KEY", "")
        if api_key:
            print("Using API key from command line or environment variable")
        correct, total = run_with_webui(
            batch, 
            model_name=args.webui_model, 
            base_url=args.webui_url,
            api_key=api_key
        )
    
    print(f"Correct: {correct}, Total: {total}, Accuracy: {correct/total:.2%}")