STARTING POINT:

    LLM is prompted to write code to interact with visual tools (
        vlm,
        grounding (returns bounding box),
        ---will add---
        sam (segmentation; the return value cannot be directly used by llm)
        google search
        ---may add---
        monocular depth estimation
    ).

    the code will be executed by an interpreter in a Jupyter notebook style,
    and added to the prompt.


    specifically, the prompt will consist of:
    1. task description
    2. tool descriptions
    3. the code that has been executed so far, paired with the output


REAL THING:
    4. strategy hint.

    the real thing is more complicated than the starting point,
    in that we will not just do a chain like mentioned, but rather a TREE,
    whose each branch is such a chain.

    the branches will be a suffix prompt for guidance (e.g. enforce the next code snippet use a certain tool, or use a certain variable),
    or termination (stop executing code snippets and report final answer).


TASK:
    this pipeline will solve vqa (visual question answering) tasks.


