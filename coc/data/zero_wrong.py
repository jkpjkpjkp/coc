from typing import Iterator, Literal
from coc.data.fulltask import FullTask
from coc.util.filter_tasks import zero_wrong

__all__ = ['zero_wrong']

if __name__ == "__main__":
    print("Testing zero_wrong function:")
    tasks = zero_wrong(offer='sub')
    count = 0
    for task in tasks:
        count += 1
        if count <= 3:
            print(f"Task {count}: {task['task_type']}")
    
    print(f"Total unsolved tasks found: {count}") 