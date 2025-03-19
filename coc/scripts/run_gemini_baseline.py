#!/usr/bin/env python3

import json
import os
from typing import Dict, List, Tuple, Iterator
from tqdm import tqdm
import pickle

from coc.data.zero import zero
from coc.data.fulltask import FullTask
from coc.tool.vqa.gemini import Gemini
from coc.util.misc import fulltask_to_task
from coc.prompts.baseline.prompt_gemini import build_trunk

def evaluate_gemini_on_zerobench() -> Tuple[int, int, Dict[str, bool]]:
    """
    Run a baseline with Gemini on ZeroBench and track which subtasks Gemini can solve.
    
    Returns:
        Tuple of (correct answers, total tasks, dictionary of {task_id: is_solved})
    """
    # Initialize Gemini
    gemini = Gemini()
    
    # Load zerobench subtasks
    tasks = list(zero(offer='sub'))
    print(f"Loaded {len(tasks)} zerobench subtasks")
    
    correct = 0
    total = 0
    solved_tasks = {}
    
    for task in tqdm(tasks):
        task_id = task['task_type']
        task_dict = fulltask_to_task(task)
        
        # Run Gemini on this task
        prompt = build_trunk(task=task_dict, offer='direct')
        try:
            response = gemini._run_multiimage(task_dict['images'], prompt)
            
            # Check if the answer is correct
            is_correct = response.lower().find(task['answer'].lower()) != -1
            
            if is_correct:
                correct += 1
                solved_tasks[task_id] = True
            else:
                solved_tasks[task_id] = False
                
            total += 1
            
            if total % 10 == 0:
                print(f"Progress: {correct}/{total} correct ({(correct/total)*100:.2f}%)")
                # Save intermediate results
                with open('data/gemini_solved_tasks.pkl', 'wb') as f:
                    pickle.dump(solved_tasks, f)
                
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            solved_tasks[task_id] = False
            total += 1
    
    # Save final results
    with open('data/gemini_solved_tasks.pkl', 'wb') as f:
        pickle.dump(solved_tasks, f)
    
    # Save human-readable results
    with open('data/gemini_solved_tasks.json', 'w') as f:
        json.dump(solved_tasks, f, indent=2)
    
    print(f"Final results: {correct}/{total} correct ({(correct/total)*100:.2f}%)")
    print(f"Results saved to data/gemini_solved_tasks.pkl and data/gemini_solved_tasks.json")
    
    return correct, total, solved_tasks

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Run evaluation
    evaluate_gemini_on_zerobench() 