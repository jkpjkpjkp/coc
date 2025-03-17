k = \
"""
{question}

setting:
you are a paranoid person and believe your own vision or any following tools is acting against you. use all sources of information, zoom in, cross-validate before answering.

These simple interfaces to grounding, vlm, sam, and depth are to get you started, but there’s a lot more. By querying the info function, you can get details, variants and detailed parameter settings.
```python
from typing import *
from PIL import Image
import numpy as np


def grounding(image: Image.Image, text: str) -> List[Tuple[int, int, int, int]]: # returns x1, y1, x2, y2
    ...

def vlm(image: Image.Image, question: str) -> str: # visual question answering
    ...

def sam(image: Image.Image) -> List[np.ndarray]: # returns H*W bool arrays (masks)
    ...

def depth(image: Image.Image) -> np.ndarray: # returns H*W uint8 array (depth)
    ...

def info(key: Literal['grounding', 'vlm', 'sam', 'depth', 'other']) -> str:
    ...

```


Here is a question for you:

You can write yython code calling them. Enclose your code:

```python
# Your code here
```

Use task['images'][0] to refer to the 1st image as PIL.Image.Image, and so on. there are {num_images} images in total.

example code and outputs (executed with same images; you can use their intermediate results in code):
{codelist}

Let's reason step by step. write code or provide final answer (\\boxed{{}}).
{hint}
"""

from coc.tool.task import Task
from typing import *
from coc.util import Pair
from coc.util.text import codes_to_str
def build_trunk(task: Task, codes: List[Pair[str]]=[], hint: str='') -> str:
    # print(task)
    question_3_dup = task['question']
    num_images = len(task['images'])
    return k.format(question=question_3_dup, num_images=num_images, codelist=codes_to_str(codes), hint='hint: ' + hint if hint else '')