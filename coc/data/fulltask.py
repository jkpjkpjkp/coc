from typing_extensions import TypedDict
from typing import List, Optional, Iterator
from PIL.Image import Image as Img
from langchain_core.tools import BaseTool

class FullTask(TypedDict):
    task_type: str
    images: List[Img]
    question: str
    answer: str
    choices: Optional[List[str]]


class TaskLoader(BaseTool):
    def inspect(self):
        print("Dataset info:")
        print(self.subtask_dataset)

        print("\nDataset features:")
        print(self.subtask_dataset.features)

        print("\nFirst few examples:")
        print(self.subtask_dataset[:3])

    def _run(self) -> Iterator[FullTask]:
        return self.convert_to_tasks(self.subtask_dataset)