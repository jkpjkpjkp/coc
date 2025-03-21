#!/usr/bin/env python3

import pickle
import os
from typing import Dict, List, Iterator, Union, Literal
from pathlib import Path

from coc.data.fulltask import FullTask
from coc.data.zero import zero, LoadZero

def load_gemini_solved_tasks(path: str = 'data/gemini_solved_tasks.pkl') -> Dict[str, bool]:
    """
    Load the dictionary of tasks solved by vanilla Gemini.
    
    Args:
        path: Path to the pickle file containing the solved tasks
        
    Returns:
        Dictionary of {task_id: is_solved}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Solved tasks file not found at {path}. Run coc/scripts/run_gemini_baseline.py first.")
    
    with open(path, 'rb') as f:
        solved_tasks = pickle.load(f)
    
    return solved_tasks

def filter_zerobench_tasks(
    include_solved: bool = False, 
    offer: str = 'sub', 
    solved_tasks_path: str = 'data/gemini_solved_tasks.pkl'
) -> Iterator[FullTask]:
    """
    Filter zerobench tasks based on whether vanilla Gemini can solve them.
    
    Args:
        include_solved: Whether to include tasks that vanilla Gemini can solve
        offer: Which zerobench split to use ('sub' or 'full')
        solved_tasks_path: Path to the pickle file containing solved tasks
        
    Returns:
        Iterator of filtered FullTask objects
    """
    # Load tasks that Gemini can solve
    solved_tasks = load_gemini_solved_tasks(solved_tasks_path)
    
    # Load all zerobench tasks
    tasks = zero(offer=offer)
    
    # Filter tasks
    for task in tasks:
        task_id = task['task_type']
        is_solved = solved_tasks.get(task_id, False)
        
        if include_solved or not is_solved:
            yield task

def zero_wrong(offer: Literal['full', 'sub'] = 'sub', solved_tasks_path: str = 'data/gemini_solved_tasks.pkl') -> Iterator[FullTask]:
    """
    Return an iterator of FullTask objects only for tasks that vanilla Gemini cannot solve.
    Provides identical interface to zero() but with filtering applied.
    
    Args:
        offer: Which zerobench split to use ('sub' or 'full')
        solved_tasks_path: Path to the pickle file containing solved tasks
        
    Returns:
        Iterator of unsolved FullTask objects
    """
    # Load tasks that Gemini can solve
    solved_tasks = load_gemini_solved_tasks(solved_tasks_path)
    
    # Use the LoadZero class directly for consistent interface
    data_loader = LoadZero(split_name='zerobench_subquestions' if offer == 'sub' else 'zerobench')
    
    # Filter tasks
    for task in data_loader.convert_to_tasks():
        task_id = task['task_type']
        is_solved = solved_tasks.get(task_id, False)
        
        if not is_solved:
            yield task

def get_task_stats(solved_tasks_path: str = 'data/gemini_solved_tasks.pkl') -> Dict[str, int]:
    """
    Get statistics about tasks solved by vanilla Gemini.
    
    Args:
        solved_tasks_path: Path to the pickle file containing solved tasks
        
    Returns:
        Dictionary with task statistics
    """
    solved_tasks = load_gemini_solved_tasks(solved_tasks_path)
    
    total_tasks = len(solved_tasks)
    solved_count = sum(1 for v in solved_tasks.values() if v)
    unsolved_count = total_tasks - solved_count
    
    return {
        'total_tasks': total_tasks,
        'solved_count': solved_count,
        'unsolved_count': unsolved_count,
        'solved_percentage': (solved_count / total_tasks) * 100 if total_tasks > 0 else 0
    }

if __name__ == "__main__":
    # Test if the file exists
    if not os.path.exists('data/gemini_solved_tasks.pkl'):
        print("Solved tasks file not found. Run coc/scripts/run_gemini_baseline.py first.")
    else:
        stats = get_task_stats()
        print(f"Zerobench task statistics:")
        print(f"  Total tasks: {stats['total_tasks']}")
        print(f"  Solved by Gemini: {stats['solved_count']} ({stats['solved_percentage']:.2f}%)")
        print(f"  Unsolved by Gemini: {stats['unsolved_count']} ({100 - stats['solved_percentage']:.2f}%)") 