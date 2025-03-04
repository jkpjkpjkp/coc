from datasets import load_dataset
from typing import Iterator
from datasets import Dataset
from coc.data.fulltask import FullTask, TaskLoader

class LoadZero(TaskLoader):
    name: str = "load_zero"
    description: str = "Load the ZeroBench dataset"
    subtask_dataset: Dataset
    def __init__(self, split_name: str = "zerobench"):
        # Load the dataset from Hugging Face
        ds = load_dataset("jonathan-roberts1/zerobench")
        super().__init__(subtask_dataset=ds[split_name])

    def convert_to_tasks(self) -> Iterator[FullTask]:
        for item in self.subtask_dataset:
            yield FullTask(
                task_type=item['question_id'],
                images=[img.convert('RGB') for img in item['question_images_decoded']],
                question=item['question_text'],
                choices=None,  # ZeroBench does not provide choices
                answer=item['question_answer']
            )
def zero() -> Iterator[FullTask]:
    return LoadZero().convert_to_tasks()

if __name__ == "__main__":
    data_loader = LoadZero()
    data_loader.inspect()

    tasks = data_loader.convert_to_tasks()
    for i, task in enumerate(tasks):
        print(task)
        if i > 2:
            break