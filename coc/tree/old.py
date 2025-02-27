from coc.tree.llm import llm, gpt4o
from coc.exec.mod import Exec
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage
from coc.prompts import system_message
from coc.data.interface import Task, Task0
from typing import Tuple
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver

from coc.util.langgraph.graph import decorate_ai_message, decorate_tool_message

def evaluate(tsk: Task, task_id: int) -> Tuple[str, bool]:
    import logging
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
                if attempt == retries - 1:
                    logging.error("All retry attempts failed.")
                    raise
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
