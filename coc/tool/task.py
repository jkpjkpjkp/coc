from typing import List, TypedDict
from PIL.Image import Image as Img


# these classes must be exact duplicates as in .context

class Bbox(TypedDict):
    """'bbox' stands for 'bounding box'"""
    box: List[float] # [x1, y1, x2, y2]
    score: float
    label: str


class Task(TypedDict):
    images: List[Img]
    question: str


TOOLSET = (
    'grounding',
    'vqa',
)