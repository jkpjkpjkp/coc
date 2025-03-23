#!/usr/bin/env python3

"""
Example script showing how to use the Hugging Face dataset of failed Gemini tasks
as a drop-in replacement for the original dataset.
"""

import os
import sys
from datasets import load_from_disk

# Ensure script can be run from anywhere
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    """
    Load and use the Hugging Face dataset of failed Gemini tasks.
    """
    # Load the dataset
    dataset_path = 'data/gemini_failed_huggingface'
    print(f"Loading Hugging Face dataset from {dataset_path}")
    
    failed_dataset = load_from_disk(dataset_path)
    test_data = failed_dataset['test']
    
    print(f"Loaded {len(test_data)} failed tasks")
    print(f"Dataset structure: {test_data}")
    print(f"Dataset fields: {test_data.column_names}")
    
    # Get all task IDs
    task_ids = test_data['task_id']
    print(f"First 10 task IDs: {task_ids[:10]}")
    
    # Example: Filter tasks based on specific criteria
    filtered_tasks = [task_id for task_id in task_ids if task_id.startswith('1_') or task_id.startswith('2_')]
    print(f"Found {len(filtered_tasks)} tasks starting with '1_' or '2_'")
    
    # Example: How to integrate with existing code that uses the zerobench dataset
    print("\nIntegration examples:")
    print("1. Using the dataset to filter zerobench tasks:")
    print("```python")
    print("from coc.data.zero import zero")
    print("all_tasks = list(zero(offer='sub'))")
    print("# Get only the failed tasks")
    print("failed_task_ids = set(test_data['task_id'])")
    print("failed_zerobench_tasks = [task for task in all_tasks if task['task_type'] in failed_task_ids]")
    print("```")
    
    print("\n2. Drop-in replacement for testing:")
    print("```python")
    print("# Instead of:")
    print("all_tasks = list(zero(offer='sub'))")
    print("# Use:")
    print("from datasets import load_from_disk")
    print("dataset = load_from_disk('data/gemini_failed_huggingface')")
    print("failed_task_ids = set(dataset['test']['task_id'])")
    print("# Then filter the original dataset or use these IDs directly")
    print("```")

if __name__ == "__main__":
    main() 