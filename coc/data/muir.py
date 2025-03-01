from datasets import load_from_disk, Dataset
from langchain_core.tools import BaseTool

from typing import Literal, Iterator
from coc.data.interface import FullTask as Task

_muir_to_Task = {
    'question': 'question',
    'options': 'choices',
    'answer': 'answer',
    'image_list': 'images',
    'task': 'task_type',
    'name': 'Muir',
    'batch': 'test',
}

class LoadMuir(BaseTool):
    name: str = 'load_muir'
    description: str = 'Load a subtask from MUIR dataset'
    subtask_dataset: Dataset
    def __init__(self, partition: Literal['Counting', 'Ordering']):
        subtask_dataset = load_from_disk('data/muir/' + partition)
        super().__init__(subtask_dataset=subtask_dataset)

    def convert_to_tasks(self) -> Iterator[Task]:
        for item in self.subtask_dataset:
            yield Task(
                task_type=item['task'],
                images=[img_jpeg.convert('RGB') for img_jpeg in item['image_list']],
                question=item['question'],
                choices=item['options'],
                answer=item['answer'],
            )
    def inspect(self):
        print('Dataset info:')
        print(self.subtask_dataset)

        print('\nDataset features:')
        print(self.subtask_dataset.features)

        print('\nFirst few examples:')
        print(self.subtask_dataset[:3])

    def _run(self):
        return self.convert_to_tasks(self.subtask_dataset)

if __name__ == '__main__':
    data_loader = LoadMuir('Ordering')
    data_loader.inspect()