#!/usr/bin/env python3

import json
import os
import sys
from typing import Dict, List, Tuple
from tqdm import tqdm

# Import datasets for Hugging Face format
try:
    from datasets import Dataset, DatasetDict
except ImportError:
    print("ERROR: 'datasets' library not found. Install with 'pip install datasets'")
    sys.exit(1)

def create_huggingface_dataset():
    """
    Create a HuggingFace dataset from the failed tasks JSON file,
    facilitating easy drop-in replacement of filtered data.
    """
    # Load the failed tasks from JSON
    failed_tasks_path = 'data/gemini_failed_tasks.json'
    if not os.path.exists(failed_tasks_path):
        print(f"Error: Could not find {failed_tasks_path}")
        sys.exit(1)
    
    # Load failed task IDs
    with open(failed_tasks_path, 'r') as f:
        failed_task_ids = json.load(f)
    
    print(f"Loaded {len(failed_task_ids)} failed task IDs from {failed_tasks_path}")
    
    # Create a list of dictionaries with task_id and failed status
    tasks = [{"task_id": task_id, "failed_by_gemini": True} for task_id in failed_task_ids]
    
    # Create dataset
    dataset = Dataset.from_list(tasks)
    dataset_dict = DatasetDict({"test": dataset})
    
    # Save to disk
    output_path = 'data/gemini_failed_huggingface'
    dataset_dict.save_to_disk(output_path)
    print(f"Saved dataset with {len(tasks)} failed task IDs to {output_path}")
    
    # Display info
    print(f"Dataset structure: {dataset}")
    print(f"Dataset fields: {dataset.column_names}")
    
    print("\nTo use this dataset, you can load it with:")
    print("from datasets import load_from_disk")
    print(f"dataset = load_from_disk('{output_path}')")
    print("test_data = dataset['test']")
    
    # Try to locate zerobench data to enhance the dataset with full task data
    print("\nLooking for zerobench data to enhance the dataset...")
    zerobench_paths = [
        'data/zerobench.json',
        'data/zero/zerobench.json',
        'submodules/zerobench/zerobench.json'
    ]
    
    for path in zerobench_paths:
        if os.path.exists(path):
            print(f"Found zerobench data at {path}")
            try:
                # Load the zerobench data
                with open(path, 'r') as f:
                    zerobench_data = json.load(f)
                
                # Filter to only include failed tasks
                failed_tasks = []
                for task in tqdm(zerobench_data, desc="Filtering tasks"):
                    if task.get('task_type') in failed_task_ids:
                        task_copy = task.copy()
                        task_copy['failed_by_gemini'] = True
                        failed_tasks.append(task_copy)
                
                if failed_tasks:
                    # Create enhanced dataset
                    enhanced_dataset = Dataset.from_list(failed_tasks)
                    enhanced_dataset_dict = DatasetDict({"test": enhanced_dataset})
                    
                    # Save enhanced dataset
                    enhanced_output_path = 'data/gemini_failed_huggingface_enhanced'
                    enhanced_dataset_dict.save_to_disk(enhanced_output_path)
                    print(f"Saved enhanced dataset with {len(failed_tasks)} tasks to {enhanced_output_path}")
                    
                    # Display info
                    print(f"Enhanced dataset structure: {enhanced_dataset}")
                    print(f"Enhanced dataset fields: {enhanced_dataset.column_names}")
                    
                    print("\nTo use the enhanced dataset, you can load it with:")
                    print("from datasets import load_from_disk")
                    print(f"dataset = load_from_disk('{enhanced_output_path}')")
                    print("test_data = dataset['test']")
                else:
                    print("Warning: No matching tasks found in zerobench data")
                
                break
            except Exception as e:
                print(f"Error processing zerobench data from {path}: {e}")
    else:
        print("Could not find zerobench data to enhance the dataset")

if __name__ == "__main__":
    create_huggingface_dataset() 