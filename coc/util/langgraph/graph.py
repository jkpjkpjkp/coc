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
import logging

def decorate_ai_message(state: MessagesState):
    """
    1. Look at the last message in `state["messages"]`.
    2. If it's an AIMessage, extract triple-backtick code blocks from .content.
    3. Create tool calls for each code block (using the tool named 'exec').
    4. Update that AIMessage's .tool_calls with the extracted code blocks.
    5. Return the updated state.
    """
    # Make sure we actually have messages:
    if "messages" not in state or not state["messages"]:
        return state

    last_message = state["messages"][-1]
    # We only decorate if the last message is an AIMessage:
    if not isinstance(last_message, AIMessage):
        return state

    # # Truncate content at the tool call end marker, for deepseek chat hallucinates code output otherwise.
    # if '<｜tool▁call▁end｜>' in last_message.content:
    #     last_message.content = last_message.content.split('<｜tool▁call▁end｜>')[0]

    # Extract code blocks enclosed by triple backticks that start with 'python'
    # Example matches:
    #   ```python
    #   print("Hello")
    #   ```
    code_blocks = re.findall(r"```python(.*?)```", last_message.content, flags=re.DOTALL)

    if len(code_blocks) == 0:
        return state

    tool_calls = []
    for block in code_blocks:
        # Skip if block only contains 'exec' (ignoring whitespace and newlines)
        if ''.join(block.split()) == 'exec':
            continue

        # Skip if block is essentially empty or just contains a final answer
        stripped_block = ''.join(block.split())
        if len(stripped_block.lower().replace('final answer', '').replace('#', '').replace(' ', '')) < 5:
            logging.warning(f"Skipping block: {block}")
            return state

        # Use a unique ID to keep track of each code block.
        call_id = str(uuid.uuid4())

        tool_calls.append(
            {
                "name": "exec",            # name of the tool to call
                "id": call_id,             # unique ID for this code call
                "args": {"code": block},   # arguments the tool needs
            }
        )
    # Create a fresh AIMessage with the same content but the newly formed tool_calls.
    decorated_message = AIMessage(
        content=last_message.content,
        tool_calls=tool_calls,
    )


    # Replace the last message in state with our newly decorated one:
    state["messages"][-1] = decorated_message
    return state

def decorate_tool_message(state: MessagesState) -> dict:
    # Make sure we actually have messages:
    if "messages" not in state or not state["messages"]:
        return state

    messages = state["messages"]
    # Find the last AIMessage and collect all subsequent ToolMessages
    last_ai_idx = None
    tool_messages = []

    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage):
            last_ai_idx = i
        elif isinstance(msg, ToolMessage) and last_ai_idx is not None:
            tool_messages.append(msg)

    if not tool_messages:
        return state

    # Merge all tool message contents into one HumanMessage
    merged_content = "\n".join(msg.content for msg in tool_messages)
    tool_message = HumanMessage(content=merged_content)

    # Get the AIMessage and merge all tool_calls
    ai_message = messages[last_ai_idx]
    merged_tool_calls = []
    if hasattr(ai_message, 'tool_calls'):
        merged_tool_calls.extend(ai_message.tool_calls)

    # Update the state: remove all tool messages and add merged message
    state["messages"] = messages[:last_ai_idx + 1]
    state["messages"].append(tool_message)
    state['messages'][-1] = tool_messages
