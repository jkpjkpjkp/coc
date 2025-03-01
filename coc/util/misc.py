from PIL.Image import Image as Img
import torch
import numpy as np
import random
from typing import Generic, TypeVar, Iterator, Iterable
import os
from coc.data.interface import FullTask


def image_to_numpy(image: Img) -> np.ndarray:
    return np.array(image.convert("RGB"))

def set_seed():
    from coc.config import SEED
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


T = TypeVar('T')

class Pair(Generic[T]):
    def __init__(self, first: T, second: T):
        self.first = first
        self.second = second

    def __getitem__(self, index: int) -> T:
        if index == 0:
            return self.first
        if index == 1:
            return self.second
        raise IndexError("Pair index out of range")

    def __len__(self) -> int:
        return 2

    def __iter__(self) -> Iterator[T]:
        yield self.first
        yield self.second

    def __eq__(self, value):
        if isinstance(value, Pair):
            return self.first == value.first and self.second == value.second
        return False

    def __str__(self):
        return str((self.first, self.second))



def generate_files(tasks: Iterable[FullTask], path: str):
    # Construct file paths
    questions_path = os.path.join(path, "questions.txt")
    answers_path = os.path.join(path, "answers.txt")

    # Open both files in write mode with UTF-8 encoding
    with open(questions_path, "w", encoding="utf-8") as qf, \
         open(answers_path, "w", encoding="utf-8") as af:
        # Iterate over tasks with indices starting at 1
        for index, task in enumerate(tasks, start=1):
            # Write the indexed question to questions.txt
            qf.write(f"{index}. {task['question']}\n")

            # If choices are present, write each choice indented with a letter
            if task['choices'] is not None:
                for i, choice in enumerate(task['choices']):
                    qf.write(f"   {chr(97 + i)}) {choice}\n")

            # Write the indexed answer to answers.txt
            af.write(f"{index}. {task['answer']}\n")