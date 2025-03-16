## code structure

coc/ and tests/ has exactly the same structure, where test file of a file is prepended with test_
logs are kept under data/log, and every module should log its io and errors in 2 files; log with timestamp.
various images under data/sample can be used for testing, e.g. onions.png is a pile of onions, and 4girls.png is 4 girls. 

## idea
in tree/ many vlm(mllm) tree-search inference strategies will be spawned, each writes code to interact with tool/
to keep local gpu at full capacity, we wrap each gpu-heavy tool into a gradio server.


## style guides
use single quote ', but use """ for google style docstring.
should use no mocking when writing tests, since this is a small project.


## trees
