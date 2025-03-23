#!/usr/bin/env python3

import json
import os
import sys
from typing import Dict, List, Tuple, Iterator
from tqdm import tqdm
import pickle
import time
import random

# Ensure script can be run from anywhere
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from coc.data.zero import zero
from coc.data.fulltask import FullTask
# Import Gemini from gromini instead of gemini
from coc.tool.vqa.gromini import Gemini
from coc.secret import GEMINI_BROKERS
from coc.util.misc import fulltask_to_task
from coc.prompts.baseline.prompt_gemini import build_trunk

def extract_rate_limited_tasks():
    """
    Extract the list of task IDs that failed due to rate limiting (429 errors)
    from the log file.
    
    Returns:
        List of task IDs that need to be retried
    """
    rate_limited_tasks = []
    
    try:
        with open('data/gemini_yanked_tasks.txt', 'r') as f:
            for line in f:
                if 'Error processing task' in line and 'Error code: 429' in line:
                    # Extract task ID from lines like: "Error processing task 5_5: Error code: 429"
                    parts = line.split('Error processing task ')
                    if len(parts) > 1:
                        task_id = parts[1].split(':')[0].strip()
                        rate_limited_tasks.append(task_id)
    except Exception as e:
        print(f"Error extracting rate limited tasks: {e}")
        
    return rate_limited_tasks

def retry_rate_limited_tasks():
    """
    Retry only the tasks that failed due to rate limiting.
    
    Returns:
        Updated dictionary of solved tasks
    """
    # Load the existing results
    solved_tasks = {}
    checkpoint_path = 'data/gemini_solved_tasks.pkl'
    
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                solved_tasks = pickle.load(f)
                print(f"Loaded checkpoint with {len(solved_tasks)} completed tasks")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            solved_tasks = {}
    
    # Also load the JSON version for better readability if needed
    json_path = 'data/gemini_solved_tasks.json'
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                json_solved = json.load(f)
                # If the pickle file failed to load, use the JSON as fallback
                if not solved_tasks:
                    solved_tasks = json_solved
                    print(f"Loaded JSON backup with {len(solved_tasks)} tasks")
        except Exception as e:
            print(f"Error loading JSON backup: {e}")
    
    # Get the list of rate-limited tasks to retry
    rate_limited_tasks = extract_rate_limited_tasks()
    print(f"Found {len(rate_limited_tasks)} rate-limited tasks to retry")
    
    if not rate_limited_tasks:
        print("No rate-limited tasks found. Exiting.")
        return solved_tasks
    
    # Initialize Gemini with broker pool - same as original
    gemini = Gemini(
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        backoff_factor=2.0,
        broker_pool=GEMINI_BROKERS,
        model_name='gemini-2.0-pro-exp-02-05'
    )
    
    # Load all zerobench subtasks
    all_tasks = list(zero(offer='sub'))
    print(f"Loaded {len(all_tasks)} zerobench subtasks")
    
    # Filter to only the rate-limited tasks that need retrying
    retry_tasks = [task for task in all_tasks if task['task_type'] in rate_limited_tasks]
    print(f"Filtered to {len(retry_tasks)} tasks that need retrying")
    
    # Process the retry tasks
    correct = sum(1 for value in solved_tasks.values() if value)
    total = len(solved_tasks)
    
    for task in tqdm(retry_tasks):
        task_id = task['task_type']
        task_dict = fulltask_to_task(task)
        
        # Run Gemini on this task
        prompt = build_trunk(task=task_dict, offer='direct')
        try:
            # Add a small random delay to prevent rate limiting
            time.sleep(random.uniform(2.0, 5.0))  # Slightly longer delays to avoid rate limits
            
            response = gemini._run_multiimage(task_dict['images'], prompt)
            
            # Check if the answer is correct
            is_correct = response.lower().find(task['answer'].lower()) != -1
            
            if is_correct:
                correct += 1
                solved_tasks[task_id] = True
                print(f"✓ Correct: {task_id}")
            else:
                solved_tasks[task_id] = False
                print(f"✗ Incorrect: {task_id}")
                print(f"  Question: {task['question']}")
                print(f"  Expected: {task['answer']}")
                print(f"  Got: {response}")
            
            # Save after each task to preserve progress
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(solved_tasks, f)
            
            with open(json_path, 'w') as f:
                json.dump(solved_tasks, f, indent=2)
                
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            # Don't mark as failed again if we get another error
            # Only overwrite if we successfully complete the task
            
            # Save after error to preserve progress
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(solved_tasks, f)
            
            with open(json_path, 'w') as f:
                json.dump(solved_tasks, f, indent=2)
    
    # Recalculate final stats
    correct = sum(1 for value in solved_tasks.values() if value)
    total = len(solved_tasks)
    print(f"Final results after retrying: {correct}/{total} correct ({(correct/total)*100:.2f}%)")
    
    return solved_tasks

def create_failed_tasks_dataset(solved_tasks):
    """
    Create a dataset containing only the tasks that Gemini genuinely failed on 
    (not those that failed due to API errors).
    
    Args:
        solved_tasks: Dictionary of {task_id: is_solved}
    """
    # Filter to only the failed tasks
    failed_tasks = {task_id: False for task_id, is_solved in solved_tasks.items() if not is_solved}
    
    # Save to JSON
    with open('data/gemini_failed_tasks.json', 'w') as f:
        json.dump(failed_tasks, f, indent=2)
    
    # Also save as pickle for convenience
    with open('data/gemini_failed_tasks.pkl', 'wb') as f:
        pickle.dump(failed_tasks, f)
    
    print(f"Created dataset of {len(failed_tasks)} failed tasks")
    print("Saved to data/gemini_failed_tasks.json and data/gemini_failed_tasks.pkl")
    
    return failed_tasks

if __name__ == "__main__":
    # Step 1: Retry rate-limited tasks
    solved_tasks = retry_rate_limited_tasks()
    
    # Step 2: Create failed tasks dataset
    failed_tasks = create_failed_tasks_dataset(solved_tasks) 