from typing import *
from PIL import Image

def grounding(image: Image.Image, text: List[str]):
    ...

def vlm(*args: Tuple[Union[Image.Image, str]]):
    ...

def sam(image: Image.Image):
    ...

def depth(image: Image.Image):
    ...

def info(text: Literal[
    'grounding',
    'vlm',
    'sam',
    'depth',
    'other',
]) -> str:
    ...