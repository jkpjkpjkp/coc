from coc.tool.task import Task
direct_answer_trunk = """
You will be presented a vqa task.
task: {}
Please answer it, and present your final answer in \\boxed{{}}. Let's think step by step.
"""

def build_trunk(task: Task):
    return direct_answer_trunk.format(task)