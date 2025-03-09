## code structure

coc/ and tests/ has exactly the same structure, where test file of a file is prepended with test_
logs are kept under data/log, and every module should log its io and errors in 2 files; log with timestamp.

## idea
in tree/ many vlm(mllm) tree-search inference strategies will be spawned, each writes code to interact with tool/
to keep local gpu at full capacity, we wrap each gpu-heavy tool into a gradio server.


## style guides
use single quote ', but use """ for google style docstring.
should use no mocking when writing tests, since this is a small project.


## current task
implement gradio servers that wraps visual models, and write unittest for them.
each server will mainly be called programatically, but should as well have a human-friendly web interface. 