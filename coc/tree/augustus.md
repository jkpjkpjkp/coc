
your task today is to implement using various tested tools in coc/tool and vlm (gemini (class Gemini in coc.tool.vqa.gemini))

to implement this strategy:

    it is a tree search, where each rollout is
    
        a simple ReACT, where action is writing code. (use class Exec in coc.exec.mod to interpret, with memory)
        but not simply writing code; rather, because Exec will be initiallized by a variant of coc.tool.context.mod, 
        it has access to visual tools (as i mentioned, they are tested and in coc/tool. )

        so, the vlm writes code to call tool, sees output and output images, then decides (write more code or finish)

    
    as you can see, all scaffolding is ready, so i want to see a clean, clear, short implementation utilising them. 

    another intern wrote coc.tree.gemi (it is tested and you may use its components), but it is combersome, and it is combersome for one reason in particular:

        it tries to implement specific methods for pre-assumed tasks.
        this is wrong and we wish to expose to vlm vanilla tools, for it to come up with strategies suited to any task given. 
        in other words, we want a minimal, elegant, general agentic strategy. 

    
    which is tough, and needs intricate design and prompting. 

        for this part, refer to coc.prompts, in particular coc/prompt_brev.md. 
        note that all prompts did not succeed.
        the common failure pattern is vlm sill not use tools well or to their full potentials.
        but please refer to them, build upon them.


in short, you should write minimal number of lines of code.

now, following TDD, please first make your design and write related unit and module tests. 