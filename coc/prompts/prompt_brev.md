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

{}

(i have repeated the question 3 times to catch your attention)

the images are stored in variables
```python
image1
image2
...
image{}
```