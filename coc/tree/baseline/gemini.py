from coc.tool.vqa import gemini
gemini = gemini._run_multiimage
from coc.data.fulltask import FullTask
from typing import List, Union, Optional, Literal, Iterable
from coc.prompts.baseline.prompt_gemini import build_trunk
from dataclasses import dataclass
from coc.exec import Exec, CONTEXT_FILE
from langchain_core.messages import AIMessage
from coc.util import Pair
from coc.util.text import extract_code, extract_boxed
from coc.tool.task import Task
from random import randint
import copy
from tqdm import tqdm
from coc.util.misc import fulltask_to_task
import textwrap
from coc.tree.une import judge_multichoice

def vlm_direct_infer(vlm, task: Task):
    return vlm(task['images'], build_trunk(task=task))

def eval_a_batch(batch: Iterable[FullTask]):
    correct = 0
    total = 0
    batch = list(batch)
    for task in tqdm(batch[::3][20:]):
        ret = vlm_direct_infer(gemini, fulltask_to_task(task))
        correct += judge_multichoice(ret, task['choices'], task['answer'])
        total += 1

        print(correct, total)

        if total - 2 * correct > 4:
            return correct, total
    return correct, total

if __name__ == '__main__':
    from coc.util.misc import set_seed
    set_seed()
    from coc.data.muir import muir
    print(eval_a_batch(muir('Ordering')))
