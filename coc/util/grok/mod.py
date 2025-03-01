import os
import sys
from coc.tree.llm import SimpleWrapper

# Initialize the ChatOpenAI instance with a custom API base

chat_model = SimpleWrapper(
    model="grok-3",  # Specify the model you want to use
    openai_api_key=os.environ['GROK_API_KEY'],  # Your OpenAI API key
    openai_api_base=os.environ['GROK_BASE_URL'],  # Custom API base URL
    temperature=0.7,
)

response = chat_model.invoke(r'''
split this into 2 prompts, one forcing continuation (code writing), another forcing final answer. ``` trunk = """ You will be presented a vqa task. (You currently do not have direct visual capabilities. ) task: {} You may access the following visual tools by writing python code to call them. Specifically, this code exactly is executed as the first cell, in what notebook your written code will dwell as following cells. ```python {} ``` concretely, this means you can use `task['images'][0]` to refer to the first image of your task. You must write Python code snippets to interact with these tools. The code will be executed, and the output will be provided back to you. Use this output to guide your reasoning. Start by formulating a plan to solve the task. You can decompose the task into smaller steps, using the result of prior code snippets to refine those steps, and ultimately using the available tools to solve the task. The code you provide in each step should be a single, self-contained snippet that can be executed by a Python interpreter. The printed output can be used to write code to execute next until you arrive at a final answer. A history of already executed code and there respective output: {} **Always enclose your code within triple backticks and specify the language as Python, like this:** ```python # Your code here ``` {} Let's think step by step. """ from typing import List, Optional from coc.util import Pair from coc.util.text import codes_to_str from coc import Task def build_trunk(task: Task, init_code_path: str, codes: List[Pair[str]], tool: Optional[str] = None): with open(init_code_path, 'r') as f: init_code = f.read() return trunk.format(task, init_code, codes_to_str(codes), tool_hint[0].format(tool) if tool else tool_hint[1])```
''')

print(response)
print(response.content)
