from coc.tool.task import Task
from typing import Literal
direct_answer_trunk = """
You will be presented a vqa task.
task: {}
Please answer it, and present your final answer in \\boxed{{}}. Let's think step by step.
"""

creativity_trunk = """
You will be presented a vqa task.
task: {}
If you can write code (and within it use any method/library you want), please brainstorm some strategies to solve this task. Let's think step by step.
"""

def build_trunk(task: Task, offer: Literal['direct', 'creativity'] = 'direct'):
    if offer == 'direct':
        return direct_answer_trunk.format(task)
    elif offer == 'creativity':
        return creativity_trunk.format(task)
    else:
        raise NotImplementedError