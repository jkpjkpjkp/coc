from coc.bots.llm import llm, gpt4o


from coc.exec.mod import Exec


from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from coc.prompts import system_message


from coc.data.interface import Task, Task0
from typing import Tuple, Annotated
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
import re, uuid
from typing import TypedDict, List
from PIL import Image

from coc.util.langgraph.graph import decorate_ai_message, decorate_tool_message

def evaluate(tsk: Task, task_id: int) -> Tuple[str, bool]:
    _answer = tsk['answer']
    tsk = Task0.from_Task(tsk)
    exec = Exec(
        '/home/jkp/hack/diane/aoa/tool_interface.py'
        '/home/jkp/hack/diane/aoa/lang/task.py'
    )
    exec.add_var('tsk', tsk)

    def assistant(state: MessagesState):
        retries = 3
        for attempt in range(retries):
            try:
                response = llm.invoke([system_message] + state['messages'])
                return {'messages': [response]}
            except Exception as e:
                logging.error(f"LLM invocation failed on attempt {attempt + 1}: {str(e)}")
                if attempt == retries - 1:  # If this was the last attempt
                    logging.error("All retry attempts failed.")
                    raise  # Re-raise the exception after all retries have been exhausted
                import time
                time.sleep((0.5, 10, 100)[attempt])

    builder = StateGraph(MessagesState)
    builder.add_node('assistant', assistant)
    builder.add_node('decorate_ai_message', decorate_ai_message)
    builder.add_node('tools', ToolNode([exec], handle_tool_errors=False))
    builder.add_node('decorate_tool_message', decorate_tool_message)
    builder.set_entry_point('assistant')
    builder.add_edge('assistant', 'decorate_ai_message')
    builder.add_conditional_edges('decorate_ai_message', tools_condition)
    builder.add_edge('tools', 'decorate_tool_message')
    builder.add_edge('decorate_tool_message', 'assistant')

    memory = MemorySaver()
    react_graph = builder.compile(checkpointer=memory)

    config = {'configurable': {'thread_id': 'une'}}
    messages = [HumanMessage(content=str(tsk))]
    messages = react_graph.invoke({'messages': messages}, config)

    return messages


def set_seed():
    seed = 42

    import torch
    torch.manual_seed(seed)

    import numpy as np
    np.random.seed(seed)

    import random
    random.seed(seed)


from tqdm import tqdm
if __name__ == '__main__':
    # print(llm.invoke('hi'))
    import logging
    # File handler - logs everything (INFO and above)
    file_handler = logging.FileHandler('evaluation9.log')
    file_handler.setLevel(logging.INFO)

    # Stream handler - logs only WARNING and above
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)

    # Basic config with format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[file_handler, stream_handler]
    )

    sys.path.append("/home/jkp/hack/diane/data")
    from load_muir import LoadMuir
    from blink import LoadBlink
    from load_zero import LoadZeroBench
    data_loader = LoadZeroBench()
    tasks = data_loader.convert_to_tasks()
    tmp = 0





    correct = 0
    incorrect = 0
    # ret = llm.invoke('hi')
    # print(ret)
    for i, tsk in tqdm(enumerate(tasks)):
        if i < tmp:
            continue
        logging.warning(f"Evaluating task {tsk}")
        ret = evaluate(tsk, i)

        logging.warning(f"Last message: {ret['messages'][-1].content}")
        logging.warning(f"choices: {tsk['choices']}")
        logging.warning(f"Task answer: {tsk['answer']}")
        logging.info(f"All messages: {ret}")

        # res = gpt4o.invoke('please tell if the answer is correct in a multichoice question. output "YES" if so, otherwise "NO".' + f"answer to be judged: {ret['messages'][-1].content}\nchoices: {tsk['choices']}\nground truth answer: {tsk['answer']}")
        res = gpt4o.invoke('please tell if the answer is correct in a vqa question. output "YES" if so, otherwise "NO".' + f"answer to be judged: {ret['messages'][-1].content}\n\nground truth answer: {tsk['answer']}")

        if 'NO' in res.content:
            incorrect += 1
        elif 'YES' in res.content:
            correct += 1
        else:
            logging.error(f"Judge returned an invalid answer: {res.content}")
        logging.error(f"correct: {correct}, incorrect: {incorrect}")
        print(correct, incorrect)

    print(correct, incorrect)
    logging.INFO(correct)
    logging.INFO(incorrect)
