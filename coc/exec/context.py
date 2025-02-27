from typing import Dict, List, Optional, TypedDict
from PIL.Image import Image as Img


class Bbox(TypedDict):
    box: List[float]
    score: float
    label: str

class Task(TypedDict):
    images: List[Img]
    question: str

from coc.tool.factory import (
    get_grounding,
    get_vqa,
)

def grounding(image: Img, objects_of_interest: Optional[List[str]] = None) -> List[Bbox]:
    return get_grounding()(image, objects_of_interest)

def vqa(image: Img, question: str) -> str:
    return get_vqa()(image, question)