from typing_extensions import TypedDict
from typing import List, Optional
from PIL.Image import Image as Img
from coc.tool.context import Task

class FullTask(TypedDict):
    task_type: str
    images: List[Img]
    question: str
    answer: str
    choices: Optional[List[str]]

    def to_task(self):
        return Task(images=self.images, question=self.question)