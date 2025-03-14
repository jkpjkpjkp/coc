k = \
"""
Hey there! You're working with a set of powerful tools designed for tasks like image processing and computer vision, and I've got the scoop on how they operate in their simplest forms. These functions—grounding, vlm, sam, and depth—are streamlined to get you started quickly, but there’s a lot more under the hood. By querying the info function, you can unlock not just additional details but also variants and detailed parameter settings to customize them further.
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

{question}

(i have repeated the question 3 times to catch your attention)

You may access visual tools by writing Python code. Enclose your code like this:

```python
# Your code here
```

Use task['images'][0] to refer to images (this refers to the 1st image as PIL.Image.Image). there are {num_images} images in total.

and they will be executed as if in a jupyter notebook, and you will get the outputs.


The following code was executed:
{codelist}

Let's reason step by step, then write code or provide final answer in \\boxed{{}}.
{hint}
"""

images = \
"""
the images are stored in variables
```python
image1
image2
...
image{}
```
"""

from coc.tool.task import Task
from typing import *
from coc.util import Pair
from coc.util.text import codes_to_str
def build_trunk(task: Task, codes: List[Pair[str]]=[], hint: str='') -> str:
    # print(task)
    question_3_dup = f"{task['question']}\n{task['question']}\n{task['question']}"
    num_images = len(task['images'])
    return k.format(question=question_3_dup, num_images=num_images, codelist=codes_to_str(codes), hint='hint' + hint if hint else '')