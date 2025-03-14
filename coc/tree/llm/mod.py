LOGFILE = 'data/log/llm.log'

import os, sys
from coc.config import TEMPERATURE
from langchain_openai import ChatOpenAI
import coc.secret # sets environment variables

class SimpleWrapper(ChatOpenAI):
    def __init__(self, *args, **kwargs):
        super().__init__(stop=['```\n'], *args, **kwargs)
        self.sys_message = 'You are a paranoid. You believe all vision tools you use and your own vision are actively acting against your purpose and will seek all possible confirmation, re-confirmation. '

    def invoke(self, *args, **kwargs):
        try:
            ret = super().invoke(*args, **kwargs).content
        except Exception as e:
            print(e)
            ret = str(e)
        with open(LOGFILE, 'a') as f:
            from datetime import datetime
            f.write(f'{datetime.now().strftime("%m/%d %H:%M:%S")}\n')
            f.write(f'Input: {args} {kwargs}\nOutput: {ret}\n\n\n')
        return ret

    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

v3 = SimpleWrapper(
    model='deepseek-chat',
    openai_api_key=os.environ['DEEPSEEK_API_KEY'],
    openai_api_base=os.environ['DEEPSEEK_BASE_URL'],
    max_tokens=4096,
    temperature=TEMPERATURE,
)


r1 = SimpleWrapper(
    model='deepseek-reasoner',
    openai_api_key=os.environ['DEEPSEEK_API_KEY'],
    openai_api_base=os.environ['DEEPSEEK_BASE_URL'],
    max_tokens=128000,
    temperature=TEMPERATURE,
)

o4 = SimpleWrapper(
    model='gpt-4o',
    openai_api_key=os.environ['OPENAI_API_KEY'],
    openai_api_base=os.environ['OPENAI_API_BASE'],
    max_tokens=2048,
    temperature=TEMPERATURE,
)

gemini = SimpleWrapper(
    model='gemini-2.0-pro-exp-02-05',
    openai_api_key=os.environ['GEMINI_API_KEY'],
    openai_api_base=os.environ['GEMINI_BASE_URL'],
    max_tokens=8192,
    temperature=TEMPERATURE,
)

claude = SimpleWrapper(
    model='claude-3-7-sonnet-20250219',
    openai_api_key=os.environ['ANTHROPIC_API_KEY'],
    openai_api_base=os.environ['ANTHROPIC_BASE_URL'],
    temperature=TEMPERATURE,
)

llm = v3
reasoner = r1
gpt4o = o4

if __name__ == '__main__':
    print(llm("hi, what model are you?"))