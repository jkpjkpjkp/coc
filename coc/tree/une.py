
MAX_DEPTH = 2

from coc.data.fulltask import FullTask
from typing import List, Union, Optional, Literal, Iterable
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

@dataclass
class Code:
    code: str
    output: str
    error: str

    def __init__(self, code: str, env: Exec):
        self.code = code
        self.output, self.error = env._run(code)

    def to_pair_of_str(self):
        return Pair(self.code, self.output)

@dataclass
class CodeList:
    _: List[Code]
    env: Exec

    def __init__(self, context: str, task: Task):
        self._ = []
        self.env = Exec(context)
        self.env.set_var('task', task)

    def append(self, code: Union[str, Code]):
        if isinstance(code, str):
            code = Code(code, self.env)
        self._.append(code)

    def to_list_of_pair_of_str(self):
        return [code.to_pair_of_str() for code in self._ if not code.error]

class Node:
    def __init__(self, codelist: CodeList, outputs: List[AIMessage]):
        self.codelist = codelist
        self.outputs = outputs

class TreeNode(Node):
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
        assert not (codes and answer), f"Both codes and answer extracted from response: {response}"
        assert codes or answer, f"Neither codes nor answer extracted from response: {response}"
        if answer:
            return answer, node
        else: # codes
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
    return None, node

def root_factory(task: Task) -> TreeNode:
    return TreeNode(
        codelist=CodeList(context=CONTEXT_FILE, task=task),
        outputs=[],
        parent=None,
        children=[],
        depth=0,
    )

def force_code_then_answer_at_each_step(task: Task, max_depth: int = MAX_DEPTH):
    from coc.tree.llm import llm
    root = root_factory(task)
    ret, node = rollout(task, root, llm, max_depth=max_depth, variant='force code')
    assert not ret, "Answer premature (force code failed)"
    ret, node = rollout(task, node, llm, max_depth=max_depth+1, variant='force answer')
    assert ret, "Answer not found (force answer failed)"
    return ret, node

def compare(output: str, answer: str):
    from coc.tool.vqa.mod import gemini_as_llm as llm
    return llm(textwrap.dedent(f'''
                compare the following two outputs and return True if they are the same (output is correct),
                    otherwise return False.
                output: {output}
                answer: {answer}'''
            )).strip() == 'True'

def judge_multichoice(output: str, choices: List[str], answer: str):
    from coc.tool.vqa.mod import gemini_as_llm as llm
    return llm(textwrap.dedent(f'''
                judge the following output and return True if, given the choices available, the party offering this output has the capability of arriving at the correct choice,
                    otherwise return False.
                output (cannot see the choices): {output}
                choices: {choices}
                answer (correct choice is): {answer}'''
            )).strip() == 'True'

def eval_a_batch(batch: Iterable[FullTask]):
    correct = 0
    total = 0
    batch = list(batch)
    for task in tqdm(batch[::3]):
        ret, node = force_code_then_answer_at_each_step(fulltask_to_task(task), MAX_DEPTH + total)
        correct += judge_multichoice(ret, task['choices'], task['answer'])
        total += 1

        print(correct, total)

        if total - 2 * correct > 4:
            return correct, total
    return correct, total

if __name__ == '__main__':
    from coc.util.misc import set_seed
    set_seed()
    from coc.data.muir import muir
    print(eval_a_batch(muir('Ordering')))
