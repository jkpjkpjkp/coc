import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from coc.config import MAX_DEPTH, BRANCH_WIDTH
from coc.tree.une import CodeList, TreeNode, root_factory
from coc.tool.task import Task
from typing import Optional, List, Iterable
from coc.util.text import extract_code, extract_boxed
from coc.prompts.prompt_troi import prompts_2nd_person
from coc.tool.context.prompt_brev import build_trunk
from coc.tool.vqa import gemini_as_llm
from coc.util.misc import fulltask_to_task, set_seed
from coc.data.fulltask import FullTask
import textwrap
from coc.util.logging import get_logger

logger = get_logger('coc', 'qua.log')

# Import Gemini for VLM support
from coc.tool.vqa.gemini import Gemini
gemini = Gemini()
def vlm(*args, **kwargs):
    return gemini.run_freestyle(*args, **kwargs)

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
    # assert not (codes and answer), f"Both code and answer extracted: {response}"
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
        answer = fulltask['answer']
        task = fulltask_to_task(fulltask)
        nodes = evaluate(task, vlm)
        
        # Get all answers for this task
        answers = [answer for _, answer in nodes]
        
        # Check if any answer is correct
        if judge_if_any(answers, answer):
            correct += 1
        total += 1

        print(correct, total)
        logger.warning(f"Correct: {correct}, Total: {total}, Accuracy: {correct/total:.2%}")
        if total - 2 * correct > 5:
            break
    
    return correct, total

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
    batch = list(zero(offer='failed'))
    print(f"Evaluating {len(batch)} tasks using Gemini VLM...")
    correct, total = eval_a_batch(batch, vlm)
    print(f"Correct: {correct}, Total: {total}, Accuracy: {correct/total:.2%}")