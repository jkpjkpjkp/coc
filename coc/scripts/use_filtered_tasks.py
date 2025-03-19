#!/usr/bin/env python3

import os
import argparse
from tqdm import tqdm

from coc.util.filter_tasks import filter_zerobench_tasks, get_task_stats
from coc.util.misc import fulltask_to_task

def main():
    """
    Example showing how to use filtered zerobench tasks that exclude
    those already solved by vanilla Gemini.
    """
    parser = argparse.ArgumentParser(description='Run tasks filtering out those solved by vanilla Gemini')
    parser.add_argument('--include-solved', action='store_true', 
                        help='Include tasks that vanilla Gemini can already solve')
    parser.add_argument('--offer', type=str, choices=['sub', 'full'], default='sub',
                        help='Which zerobench split to use (default: sub)')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only show statistics, don\'t process tasks')
    
    args = parser.parse_args()
    
    # Display stats
    if not os.path.exists('data/gemini_solved_tasks.pkl'):
        print("No stats available. Run coc/scripts/run_gemini_baseline.py first to generate baseline data.")
        return
    
    stats = get_task_stats()
    print(f"Zerobench task statistics:")
    print(f"  Total tasks: {stats['total_tasks']}")
    print(f"  Solved by Gemini: {stats['solved_count']} ({stats['solved_percentage']:.2f}%)")
    print(f"  Unsolved by Gemini: {stats['unsolved_count']} ({100 - stats['solved_percentage']:.2f}%)")
    
    if args.stats_only:
        return
    
    # Get filtered tasks
    tasks = list(filter_zerobench_tasks(
        include_solved=args.include_solved,
        offer=args.offer
    ))
    
    print(f"\nUsing filtered tasks ({len(tasks)} tasks):")
    print(f"  include_solved: {args.include_solved}")
    print(f"  offer: {args.offer}")
    
    # Process tasks (demo)
    for i, task in enumerate(tqdm(tasks[:5])):  # Just process first 5 for demo
        task_dict = fulltask_to_task(task)
        print(f"\nTask {i+1}/{len(tasks)}:")
        print(f"  ID: {task['task_type']}")
        print(f"  Question: {task['question']}")
        print(f"  # Images: {len(task['images'])}")
        
        # Your model processing would go here
        # Example:
        # response = your_model.process(task_dict['images'], task_dict['question'])
        # compare_with_answer(response, task['answer'])
    
    print("\nDemo complete. In a real evaluation, you would process all tasks with your model.")

if __name__ == "__main__":
    main() 