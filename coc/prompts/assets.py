head = '''
You will be presented a vqa task.
You currently do not have direct visual capabilities, but you will have access to powerful visual tools.
You will interact with them by writing python code calling them.
'''


bbox = '''

from dataclasses import dataclass
@dataclass
class Bbox:
    box: List[float]
    score: float
    label: str
'''


task = '''

class Task(TypedDict):
    task_type: str
    images: List[Image.Image]
    question: str
    choices: List[str]

tsk: Task
'''

task_with_choices = '''

class Task(TypedDict):
    task_type: str
    images: List[Image.Image]
    question: str
    choices: List[str]

tsk: Task
'''



py = '```python\n'
yp = '```\n'