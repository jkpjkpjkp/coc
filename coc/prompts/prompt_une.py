tool_hint = (
    "you should consider using {} tools. ",
    "you should think of something out of the box and don't simply use tools. "
)



trunk = """
    You will be presented a vqa task.
    (You currently do not have direct visual capabilities. )

    task:
    {}

    You may access the following visual tools by writing python code to call them.
    Specifically, this code exactly is executed as the first cell, in what notebook your written code will dwell as following cells.

    ```python
    {}
    ```

    concretely, this means you can use `task['images'][0]` to refer to the first image of your task.

    You must write Python code snippets to interact with these tools. The code will be executed, and the output will be provided back to you. Use this output to guide your reasoning.

    Start by formulating a plan to solve the task. You can decompose the task into smaller steps, using the result of prior code snippets to refine those steps, and ultimately using the available tools to solve the task.

    The code you provide in each step should be a single, self-contained snippet that can be executed by a Python interpreter.
    The printed output can be used to write code to execute next until you arrive at a final answer.

    A history of already executed code and there respective output:

    {}

    **Always enclose your code within triple backticks and specify the language as Python, like this:**

    ```python
    # Your code here
    ```

    {}

    Let's think step by step.
"""

from typing import List, Optional, Literal
from coc.util import Pair
from coc.util.text import codes_to_str
from coc.tool.task import Task





# Code Continuation Prompt (forces code writing)
code_continuation_trunk = """
You will be presented a vqa task. (You currently do not have direct visual capabilities.)
task: {}
You may access visual tools by writing Python code. The following code is executed first:
```python
{}
```

Use task['images'][0] to reference images. Write self-contained Python snippets that can be executed. Previous code/output history:
{}
**Always enclose your code within triple backticks and specify the language as Python, like this:**

```python
# Your code here
```

Continue analyzing and write NEW CODE to progress toward solving the task.

{}

(however, if you are absolutely sure of a final answer, write no mode code and present your final answer after "Final Answer:".)

Let's think step by step.
"""

final_answer_trunk = """
You will be presented a vqa task. (You currently do not have direct visual capabilities.)
task: {}
The following code was executed to gather information:

{}{}{}
Based on these outputs, provide the FINAL ANSWER to the task. Do NOT write more code. Present your final answer in \boxed{{}}. Let's think step by step.
"""

def build_trunk(task: Task, init_code_path: str, codes: List[Pair[str]], tool: Optional[str] = None, variant: Literal['neutral', 'force code', 'force answer'] = 'neutral'):
    with open(init_code_path, 'r') as f:
        init_code = f.read()
    if variant == 'neutral':
        t = trunk
    elif variant == 'force code':
        t = code_continuation_trunk
    elif variant == 'force answer':
        t = final_answer_trunk
        tool = None
        init_code=''
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return t.format(task, init_code, codes_to_str(codes), tool_hint[0].format(tool) if tool else tool_hint[1])