"""only code.

since only code returns visual information, this version (still a chain) retains only code and its output.
"""

from coc.config import MAX_DEPTH

from coc.data.fulltask import FullTask
from typing import *
from coc.prompts.prompt_une import build_trunk
from dataclasses import dataclass
from coc.exec import Exec, CONTEXT_FILE
from langchain_core.messages import AIMessage
from coc.util import Pair
from coc.util.text import extract_code, extract_boxed
from coc.tool.task import Task, TOOLSET
from random import randint
import copy
from tqdm import tqdm
from coc.util.misc import fulltask_to_task
import textwrap
import copy
from typing import List, Union
from PIL import Image

@dataclass
class Code:
    """a code cell.

    paired with its output. """
    code: str
    output: str
    error: str
    output_images: List[Image.Image]

    def __init__(self, code: str, env: Exec):
        self.code = code
        self.output, self.error, self.output_images = env._run(code)

    def to_pair_of_str(self):
        return Pair(self.code, self.output)

    def to_human_message(self):
        raise NotImplementedError


@dataclass
class CodeList:
    """a ipynb notebook.

    env is after executing sequentially all codeblocks.
    """
    _: List[Code]
    env: Exec

    def __init__(self, context: str, task: Task):
        self._ = []
        self.env = Exec(CONTEXT_FILE)
        self.env.set_var('task', task)

    def append(self, code: Union[str, Code]):
        if isinstance(code, str):
            code = Code(code, self.env)
        self._.append(code)

    def to_list_of_pair_of_str(self):
        """main purpose is to filter out code with error.

        current strat is to ignore all code with empty output.
        future may need to improve, when added display().
        """
        return [code.to_pair_of_str() for code in self._ if code.output]

    def deepcopy(self):
        # Create a new instance
        result = CodeList(context="", task=None)  # Create with dummy values

        # Deep copy all attributes
        result._ = copy.deepcopy(self._)
        result.env = copy.deepcopy(self.env)
        
        # Ensure image variables are properly copied
        task = self.env.get_var('task')
        if task and 'images' in task:
            for i, img in enumerate(task['images']):
                result.env.set_var(f'image_{i+1}', img)

        return result

    def visualize_all_images(self) -> List[Tuple[str, Image.Image]]:
        ret = []
        for k, v in self.env.globals.items():
            if isinstance(v, Image.Image):
                ret.append((k, v))
        return ret


class Node:
    """a ipynb and raw llm outputs in each iteration. """
    def __init__(self, codelist: CodeList, outputs: List[AIMessage]):
        self.codelist = codelist
        self.outputs = outputs


class TreeNode(Node):
    """a Node, but in a tree.

    aka have children and parent, and other tree statistics (e.g. depth).
    """
    def __init__(self, codelist: CodeList, outputs: List[AIMessage], parent: Optional['TreeNode'], children: List['TreeNode'], depth: int):
        super().__init__(codelist, outputs)
        self.parent = parent
        self.children = children
        self.depth = depth

    def to_list(self) -> List['TreeNode']:
        """creates a list from root to this node. """
        if self.parent is None:
            return [self]
        else:
            return self.parent.to_list() + [self]


def rollout(task: Task, node: TreeNode, llm, max_depth: int = MAX_DEPTH, variant: Literal['neutral', 'force code', 'force answer'] = 'neutral'):
    """rollout a chain.

    Args:
        variant: prompt variant. e.g. 'force code' will encourage outputing code and not final answer.
    """
    while node.depth <= max_depth:
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
            return answer, node
        elif codes:  # Only process codes if we have them
            child_codelist = copy.deepcopy(node.codelist)
            for code in codes:
                child_codelist.append(code)
            child = TreeNode(
                codelist=child_codelist,
                outputs=node.outputs + [response],
                parent=node,
                children=[],
                depth=node.depth + 1,
            )
            node.children.append(child)
            node = child
        else:
            # If we have neither codes nor answer, continue with current node
            continue

    return None, node


def root_factory(task: Task) -> TreeNode:
    """returns a new root. """
    codelist = CodeList(context=CONTEXT_FILE, task=task)
    
    # Ensure image variables are set up correctly in the root node's environment
    if 'images' in task:
        for i, img in enumerate(task['images']):
            codelist.env.set_var(f'image_{i+1}', img)
    
    return TreeNode(
        codelist=codelist,
        outputs=[],
        parent=None,
        children=[],
        depth=0,
    )


def force_code_then_answer_at_each_step(task: Task, max_depth: int = MAX_DEPTH):
    from coc.tree.llm import llm
    root = root_factory(task)
    ret, node = rollout(task, root, llm, max_depth=max_depth, variant='force code')
    if ret:  # If we got an answer in force code mode, return it
        return ret, node
    ret, node = rollout(task, node, llm, max_depth=max_depth+1, variant='force answer')
    if not ret:  # If we still don't have an answer, something went wrong
        raise ValueError("No answer found after force answer mode")
    return ret, node


def compare(output: str, answer: str):
    from coc.tool.vqa import gemini_as_llm as llm
    ret = llm(textwrap.dedent(f'''
                compare the following two outputs and return True if they are the same (output is correct),
                    otherwise return False.
                output: {output}
                answer: {answer}'''
            ))
    return 'False' not in ret and 'True' in ret

def judge_multichoice(output: str, choices: List[str], answer: str):
    from coc.tool.vqa import gemini_as_llm as llm
    ret = llm(textwrap.dedent(f'''
                judge the following output and return True if, given the choices available, the party offering this output has the capability of arriving at the correct choice,
                    otherwise return False.
                output (cannot see the choices): {output}
                choices: {choices}
                answer (correct choice is): {answer}'''
            ))
    return ret.strip().lower() == 'true'


def eval_a_batch(batch: Iterable[FullTask]):
    correct = 0
    total = 0
    batch = list(batch)
    for i, task in tqdm(enumerate(batch)):
        ret, node = force_code_then_answer_at_each_step(fulltask_to_task(task), MAX_DEPTH)
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
    print(eval_a_batch(zero(offer='sub')))