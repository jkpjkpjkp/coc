from datasets import load_from_disk, Dataset
from langchain_core.tools import BaseTool
from pathlib import Path
from typing import Literal, Iterator, Optional
from coc.data.fulltask import FullTask

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

    def convert_to_tasks(self) -> Iterator[FullTask]:
        for item in self.subtask_dataset:
            yield FullTask(
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

def muir(partition: Literal['Counting', 'Ordering']):
    return LoadMuir(partition).convert_to_tasks()

class MuirToMarkdown(BaseTool):
    name: str = 'muir_to_markdown'
    description: str = 'Convert MUIR tasks to markdown format'
    partition: Literal['Counting', 'Ordering']
    output_path: str
    
    def __init__(self, partition: Literal['Counting', 'Ordering'], output_path: Optional[str] = None):
        super().__init__()
        self.partition = partition
        self.output_path = output_path or f'muir_{partition.lower()}.md'
        
    def _format_task(self, task: FullTask) -> str:
        """Format a single task into markdown"""
        return f"""## {task.task_type}

**Question:** {task.question}

**Options:**
{'\n'.join(f'- {opt}' for opt in task.choices)}

**Answer:** {task.answer}

---

"""
    
    def _run(self):
        """Write all tasks to markdown file"""
        loader = LoadMuir(self.partition)
        output = Path(self.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with output.open('w', encoding='utf-8') as f:
            for task in loader.convert_to_tasks():
                f.write(self._format_task(task))
        
        return f"Successfully wrote {len(tasks)} tasks to {self.output_path}"

def muir_to_markdown(partition: Literal['Counting', 'Ordering'], output_path: Optional[str] = None):
    """Convert MUIR tasks to markdown format"""
    return MuirToMarkdown(partition, output_path)._run()

if __name__ == '__main__':
    # Example usage
    muir_to_markdown('Ordering', 'muir_ordering.md')
