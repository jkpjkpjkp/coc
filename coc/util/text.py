import re
from typing import List
from coc.util import Pair
import textwrap

def extract_code(llm_response: str) -> List[str]:
    """Extracts code block from the LLM response.

    since we set stop=['```\n'], there will only be atmost one.
    and it (or None) will be returned.
    """
    matches = re.findall(r"```python\n(.*?)(?:\n```|$)", llm_response, re.DOTALL)
    # remove empty blocks, that ontains only english, or only comments
    matches = [match for match in matches if not re.match(r"^\s*#", match) and not re.match(r"^[a-zA-Z]", match)]

    # Strip leading/trailing whitespace from each code block while preserving internal indentation
    return [textwrap.dedent(match) for match in matches]

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
