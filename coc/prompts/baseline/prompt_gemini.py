from coc.tool.task import Task
from typing import Literal, Dict

def build_trunk(task: Dict, offer: Literal['direct', 'task']):
    """
    Builds a prompt for the Gemini model for ZeroBench tasks.
    
    Args:
        task: A dictionary containing the task details including 'question'
        offer: A string indicating the type of prompt to build:
               - 'direct': A direct question prompt for Gemini
               - 'task': A more structured task prompt
    
    Returns:
        A string containing the formatted prompt
    """
    if offer == 'direct':
        # Simple direct prompt that just uses the question as is
        return task['question']
    else:
        # More structured prompt with specific instructions
        return f"""Look at the provided images and answer the following question:

Question: {task['question']}

Provide a short, accurate answer. If asked to identify, name, or describe something in the image, be specific and precise in your response.
"""