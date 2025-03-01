from coc import Task
from typing import List, Union, Optional
from coc.prompts.prompt_une import build_trunk
from dataclasses import dataclass
from coc.exec import Exec, CONTEXT_FILE
from langchain_core.messages import AIMessage
from coc.util import Pair
from coc.util.text import extract_code, extract_boxed
from coc.tool import TOOLSET
from random import randint
import copy

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

    def __init__(self, context: str):
        self._ = []
        self.env = Exec(context)

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

def rollout(task: Task, node: TreeNode, llm, max_depth: int = 10):
    current_node = node
    while current_node.depth <= max_depth:
        message = build_trunk(
            task=task,
            init_code_path=CONTEXT_FILE,
            codes=current_node.codelist.to_list_of_pair_of_str(),
            tool=TOOLSET[randint(0, len(TOOLSET)-1)] if TOOLSET else None,
        )
        response = llm(message)
        codes = extract_code(response)
        answer = extract_boxed(response)
        assert not (codes and answer), f"Both codes and answer extracted from response: {response}"
        if answer is not None:
            return answer, node
        elif codes:
            child_codelist = copy.deepcopy(current_node.codelist)
            for code in codes:
                child_codelist.append(code)
            child = TreeNode(
                codelist=child_codelist,
                outputs=current_node.outputs + [response],
                parent=current_node,
                children=[],
                depth=current_node.depth + 1,
            )
            current_node.children.append(child)
            current_node = child
        else:
            return None
    return None

def judgement(answers, trajs, reasoner):
    reasoner

def eval(task: Task, llm, max_depth: int = 10, max_attempts: int = 10):
    root = TreeNode(
        codelist=CodeList(context=CONTEXT_FILE),
        outputs=[],
        parent=None,
        children=[],
        depth=0,
    )

    answers = []
    trajs = []
    for _ in range(max_attempts):
        ret = rollout(task, root, llm, max_depth)
        if ret:
            answers.append(ret[0])
            trajs.append(ret[1])



    return None