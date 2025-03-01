import re
from typing import List
from coc.util import Pair

def extract_code(llm_response: str) -> List[str]:
    """Extracts code blocks from the LLM response, returning them as a list of strings."""
    matches = re.findall(r"```python\n(.*?)```", llm_response, re.DOTALL)
    code_blocks = [match.strip() for match in matches]
    return code_blocks

def extract_boxed(llm_response: str) -> str:
    """Extracts boxed text from the LLM response, ensures there is only one (or none), and returns it (or '')."""
    matches = re.findall(r"\\boxed{(.*?)}", llm_response)
    assert len(matches) <= 1, f"Multiple boxed texts found in the response {llm_response}."
    if len(matches) == 1:
        return matches[0]
    else:
        return ""

def codes_to_str(codes: List[Pair[str]]) -> str:
    return '\n\n'.join([f'''
    ```python
    {x.first}
    ```

    output:
    {x.second}
    ''' for x in codes])