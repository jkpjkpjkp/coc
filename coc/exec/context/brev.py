from typing import *
from PIL import Image
import numpy as np


def grounding(image: Image.Image, text: str) -> List[Tuple[int, int, int, int]]: # returns x1, y1, x2, y2
    ...

def vlm(x: List[Union[Image.Image, str]]) -> str:
    ...

def sam(image: Image.Image) -> List[np.ndarray]: # returns list of H*W bool array (masks)
    ...

def depth(image: Image.Image) -> np.ndarray: # returns H*W uint8 array (depth)
    ...

def info(key: Literal['grounding', 'vlm', 'sam', 'depth', 'other']) -> str:
    ...
