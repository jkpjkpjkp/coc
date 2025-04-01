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
from coc.tool.vqa.gemini import Gemini, GEMINI_BROKERS
from coc.util.misc import fulltask_to_task
from coc.prompts.baseline.prompt_gemini import build_trunk

def evaluate_gemini_on_zerobench() -> Tuple[int, int, Dict[str, bool]]:
    """
    Run a baseline with Gemini on ZeroBench and track which subtasks Gemini can solve.
    Uses a pool of Gemini brokers for better reliability.
    
    Returns:
        Tuple of (correct answers, total tasks, dictionary of {task_id: is_solved})
    """
    # Display available brokers
    print(f"Available Gemini brokers: {len(GEMINI_BROKERS)}")
    for i, broker in enumerate(GEMINI_BROKERS):
        api_key, base_url = broker
        # Mask most of the API key for security
        masked_key = api_key[:8] + "..." + api_key[-4:]
        print(f"Broker {i+1}: URL={base_url}, API Key={masked_key}")
    
    # Initialize Gemini with broker pool
    gemini = Gemini(
        max_retries=3,  # Retries for each broker
        base_delay=1.0,
        max_delay=30.0,
        backoff_factor=2.0,
        broker_pool=GEMINI_BROKERS,
        model_name='gemini-2.0-pro-exp-02-05'
    )
    
    # Create output directory
    os.makedirs('data', exist_ok=True)
    
    # Load zerobench subtasks
    tasks = list(zero(offer='sub'))
    print(f"Loaded {len(tasks)} zerobench subtasks")
    
    # Check if we have a checkpoint to continue from
    checkpoint_path = 'data/gemini_solved_tasks.pkl'
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                solved_tasks = pickle.load(f)
                print(f"Loaded checkpoint with {len(solved_tasks)} completed tasks")
                
                # Count correct answers from checkpoint
                correct = sum(1 for value in solved_tasks.values() if value)
                total = len(solved_tasks)
                print(f"Checkpoint stats: {correct}/{total} correct ({(correct/total)*100:.2f}%)")
                
                # Filter out tasks that have already been processed
                tasks = [task for task in tasks if task['task_type'] not in solved_tasks]
                print(f"Remaining tasks to process: {len(tasks)}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch...")
            solved_tasks = {}
            correct = 0
            total = 0
    else:
        # Start from scratch
        solved_tasks = {}
        correct = 0
        total = 0
    
    # Process remaining tasks
    for task in tqdm(tasks):
        task_id = task['task_type']
        task_dict = fulltask_to_task(task)
        
        # Run Gemini on this task
        prompt = build_trunk(task=task_dict, offer='direct')
        try:
            # Add a small random delay to prevent rate limiting
            time.sleep(random.uniform(0.5, 2.0))
            
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
                
            total += 1
            
            if total % 5 == 0:
                print(f"Progress: {correct}/{total} correct ({(correct/total)*100:.2f}%)")
                # Save intermediate results
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(solved_tasks, f)
                
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            solved_tasks[task_id] = False
            total += 1
            
            # Save after error to preserve progress
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(solved_tasks, f)
    
    # Save final results
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(solved_tasks, f)
    
    # Save human-readable results
    with open('data/gemini_solved_tasks.json', 'w') as f:
        json.dump(solved_tasks, f, indent=2)
    
    print(f"Final results: {correct}/{total} correct ({(correct/total)*100:.2f}%)")
    print(f"Results saved to data/gemini_solved_tasks.pkl and data/gemini_solved_tasks.json")
    
    return correct, total, solved_tasks

if __name__ == "__main__":
    # Run evaluation
    evaluate_gemini_on_zerobench()