from datasets import load_dataset
from typing import Iterator, Literal
from datasets import Dataset
from coc.data.fulltask import FullTask, TaskLoader
import json
import os

class LoadZero(TaskLoader):
    name: str = "load_zero"
    description: str = "Load the ZeroBench dataset"
    subtask_dataset: Dataset
    split_name: str
    def __init__(self, split_name: str = "zerobench", filter_failed: bool = False):
        if split_name == "filtered":
            # Use the complete dataset instead of the minimal one
            complete_dataset_path = "/home/jkp/hack/coc/data/gemini_failed_huggingface_complete"
            if os.path.exists(complete_dataset_path):
                ds = load_dataset(complete_dataset_path)["test"]
                print(f"Loaded complete filtered dataset with {len(ds)} tasks")
            else:
                # Fallback to minimal dataset
                ds = load_dataset("/home/jkp/hack/coc/data/gemini_failed_huggingface")["test"]
                print(f"Loaded minimal filtered dataset with {len(ds)} tasks (complete dataset not found)")
            super().__init__(subtask_dataset=ds, split_name=split_name)
        else:
            # Load the dataset from Hugging Face
            ds = load_dataset("jonathan-roberts1/zerobench")
            
            # If filtering to failed tasks, apply the filter
            if filter_failed:
                # Load the failed task IDs from JSON
                failed_tasks_path = '/home/jkp/hack/coc/data/gemini_failed_tasks.json'
                if os.path.exists(failed_tasks_path):
                    with open(failed_tasks_path, 'r') as f:
                        failed_task_ids = set(json.load(f).keys())
                    
                    # Define a filter function
                    def is_failed_task(example):
                        return example['question_id'] in failed_task_ids
                    
                    # Filter the dataset
                    ds_filtered = ds[split_name].filter(is_failed_task)
                    print(f"Filtered to {len(ds_filtered)} failed tasks out of {len(ds[split_name])}")
                    super().__init__(subtask_dataset=ds_filtered, split_name=split_name)
                    return
            
            super().__init__(subtask_dataset=ds[split_name], split_name=split_name)

    def convert_to_tasks(self) -> Iterator[FullTask]:
        # Simplified implementation - always use question_id regardless of split_name
        for item in self.subtask_dataset:
            yield FullTask(
                task_type=item['question_id'],
                images=[img.convert('RGB') for img in item['question_images_decoded']],
                question=item['question_text'],
                choices=None,  # ZeroBench does not provide choices
                answer=item['question_answer']
            )

def zero(offer: Literal['full', 'sub', 'failed'] = 'sub') -> Iterator[FullTask]:
    """
    Load ZeroBench dataset tasks.
    
    Args:
        offer: Options for loading the dataset
            - 'full': Load full tasks
            - 'sub': Load subtasks
            - 'failed': Load only the subtasks that Gemini failed on
            
    Returns:
        Iterator of FullTask objects
    """
    if offer == 'failed':
        # Load subtasks but filter to only include failed tasks
        return LoadZero(split_name='zerobench_subquestions', filter_failed=True).convert_to_tasks()
    else:
        # Original behavior
        return LoadZero(split_name='zerobench_subquestions' if offer == 'sub' else 'zerobench').convert_to_tasks()

if __name__ == "__main__":
    # Example of filtering to failed tasks
    print("Testing the 'failed' option:")
    tasks = list(zero(offer='failed'))
    print(f"Loaded {len(tasks)} failed tasks")
    
    # # Example of using the filtered dataset directly
    # print("\nTesting the 'filtered' option:")
    # data_loader = LoadZero(split_name="filtered")
    # tasks = list(data_loader.convert_to_tasks())
    # print(f"Loaded {len(tasks)} tasks from the filtered dataset")
    
    for i, task in enumerate(tasks[:3]):
        print(f"Task {i+1}: {task['task_type']}")
        print(f"  Question: {task['question']}")
        print(f"  Answer: {task['answer']}")
        if i >= 2:
            break