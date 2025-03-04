from coc.tree.une import (
    MAX_DEPTH,
    FullTask,
    Code,
    CodeList,
    Node,
    TreeNode,
    root_factory,
    compare,
    judge_multichoice,
    fulltask_to_task
)
from typing import List, Optional, Literal, Iterable
from coc.prompts.prompt_une import build_trunk
from coc.exec import Exec, CONTEXT_FILE
from langchain_core.messages import AIMessage
from coc.util import Pair
from coc.util.text import extract_code, extract_boxed
from coc.tool.task import Task, TOOLSET
from random import randint
import copy
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_diversity(samples: List[str]) -> np.ndarray:
    """Calculate pairwise diversity scores using TF-IDF and cosine similarity"""
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(samples)
    similarity = cosine_similarity(tfidf)
    # Convert similarity to diversity (1 - similarity)
    return 1 - similarity

def select_diverse_nodes(nodes: List[TreeNode], n: int) -> List[TreeNode]:
    """Select n most diverse nodes based on their code outputs"""
    if len(nodes) <= n:
        return nodes

    # Get all code outputs as strings
    outputs = []
    for node in nodes:
        outputs.append("\n".join([code.output for code in node.codelist._]))

    # Calculate diversity matrix
    diversity = calculate_diversity(outputs)

    # Greedy selection of diverse nodes
    selected = []
    remaining = set(range(len(nodes)))

    # Start with most diverse pair
    i, j = np.unravel_index(np.argmax(diversity), diversity.shape)
    selected.append(i)
    selected.append(j)
    remaining.remove(i)
    remaining.remove(j)

    # Add remaining nodes that maximize diversity
    while len(selected) < n and remaining:
        # Calculate average diversity to selected nodes
        avg_diversity = np.mean(diversity[list(remaining), :][:, selected], axis=1)
        # Select node with highest average diversity
        next_idx = list(remaining)[np.argmax(avg_diversity)]
        selected.append(next_idx)
        remaining.remove(next_idx)

    return [nodes[i] for i in selected]

def rollout_diverse(
    task: Task,
    node: TreeNode,
    llm,
    n1: int = 5,
    n2: int = 2,
    max_depth: int = MAX_DEPTH,
    variant: Literal['neutral', 'force code', 'force answer'] = 'neutral'
):
    """Rollout with diversity-based branching"""
    if node.depth > max_depth:
        return None, node

    # Generate N1 candidate responses
    candidates = []
    for _ in range(n1):
        message = build_trunk(
            task=node.codelist.env.get_var('task'),
            init_code_path=CONTEXT_FILE,
            codes=node.codelist.to_list_of_pair_of_str(),
            tool=TOOLSET[randint(0, len(TOOLSET)-1)] if TOOLSET else None,
            variant=variant
        )
        response = llm(message)
        codes = extract_code(response)
        answer = extract_boxed(response)

        if answer:
            # If we get an answer, return immediately
            return answer, node

        # Create candidate node
        child_codelist = copy.deepcopy(node.codelist)
        for code in codes:
            child_codelist.append(code)
        candidate = TreeNode(
            codelist=child_codelist,
            outputs=node.outputs + [response],
            parent=node,
            children=[],
            depth=node.depth + 1,
        )
        candidates.append(candidate)

    # Select N2 most diverse candidates to keep
    selected = select_diverse_nodes(candidates, n2)

    # Add selected nodes as children
    node.children.extend(selected)

    # Continue rollout from each selected node
    for child in selected:
        result, final_node = rollout_diverse(task, child, llm, n1, n2, max_depth, variant)
        if result:
            return result, final_node

    return None, node

def force_code_then_answer_with_diversity(
    task: Task,
    n1: int = 5,
    n2: int = 2,
    max_depth: int = MAX_DEPTH
):
    """Force code then answer with diversity-based branching"""
    from coc.tree.llm import llm
    root = root_factory(task)

    # First phase: force code exploration
    ret, node = rollout_diverse(
        task, root, llm,
        n1=n1, n2=n2,
        max_depth=max_depth,
        variant='force code'
    )
    assert not ret, "Answer premature (force code failed)"

    # Second phase: force answer
    ret, node = rollout_diverse(
        task, node, llm,
        n1=n1, n2=n2,
        max_depth=max_depth+1,
        variant='force answer'
    )
    assert ret, "Answer not found (force answer failed)"

    return ret, node

def eval_a_batch_with_diversity(batch: Iterable[FullTask], n1: int = 5, n2: int = 2):
    """Evaluate batch with diversity-based search"""
    correct = 0
    total = 0
    batch = list(batch)

    for i, task in tqdm(enumerate(batch[::3][18:])):
        ret, node = force_code_then_answer_with_diversity(
            fulltask_to_task(task),
            n1=n1,
            n2=n2,
            max_depth=MAX_DEPTH
        )
        correct += judge_multichoice(ret, task['choices'], task['answer'])
        total += 1

        print(correct, total)

        if total - 2 * correct > 4:
            return correct, total
    return correct, total

if __name__ == '__main__':
    from coc.util.misc import set_seed
    set_seed()
    from coc.data.zero import zero
    print(eval_a_batch_with_diversity(zero()))
