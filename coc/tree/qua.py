"""vlm version.

based on tro.
"""



import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from coc.config import MAX_DEPTH, BRANCH_WIDTH
from coc.tree.une import CodeList, TreeNode, root_factory
from coc.tool.task import Task
from typing import Optional, List, Iterable
from coc.util.text import extract_code, extract_boxed
from coc.prompts.prompt_troi import prompts_2nd_person
from coc.exec.context.prompt_brev import build_trunk
from coc.tool.vqa import gemini_as_llm
from coc.util.misc import fulltask_to_task, set_seed
from coc.data.fulltask import FullTask
import textwrap

def generate_one_child(parent: TreeNode, suggestive_hint: str, vlm) -> tuple[TreeNode, Optional[str]]:
    """
    Generate a single child node from a parent using a suggestive prompt.

    Args:
        parent: Parent node to expand from.
        suggestive_hint: Prompt to guide the AI's response.
        llm: Language model function.

    Returns:
        Tuple of (child node, answer if found, else None).
    """
    message = build_trunk(
        task=parent.codelist.env.get_var('task'),
        codes=parent.codelist.to_list_of_pair_of_str(),
        # images=parent.codelist.env.get_var('images'),
        hint=suggestive_hint,
    )
    response = vlm(trunk=c)
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