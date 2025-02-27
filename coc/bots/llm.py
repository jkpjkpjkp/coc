import os, sys
os.environ['DEEPSEEK_API_KEY'] = 'sk-d1ff17b2b90a47d682453328f92cdc5f'
os.environ['DEEPSEEK_BASE_URL'] = 'https://api.deepseek.com'


os.environ['LANGCHAIN_API_KEY'] = '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'
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['LANGSMITH_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGSMITH_API_KEY'] = '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'
os.environ['LANGSMITH_PROJECT'] = 'zerobench'



os.environ['OPENAI_API_KEY'] = 'sk-aDjzEoivIbVx4o9xLIEUDrpRaNTDOOhW1rTPhCsGsdjTa3Or'
os.environ['OPENAI_API_BASE'] = 'https://chatapi.littlewheat.com/v1'

os.environ['OPEN_ROUTER_API_KEY'] = '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'
os.environ['OPEN_ROUTER_API_BASE'] = 'https://openrouter.ai/api/v1'

from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(
#     model=('deepseek-chat', 'deepseek-reasoner')[0],
#     openai_api_key=os.environ['DEEPSEEK_API_KEY'],
#     openai_api_base=os.environ['DEEPSEEK_BASE_URL'],
#     max_tokens=2048,
#     temperature=0,
# )

llm = ChatOpenAI(
    model='deepseek/deepseek-chat',
    openai_api_key=os.environ['OPEN_ROUTER_API_KEY'],
    openai_api_base=os.environ['OPEN_ROUTER_API_BASE'],
    max_tokens=4096,
    temperature=0,
)

gpt4o = ChatOpenAI(
    model='gpt-4o',
    openai_api_key=os.environ['OPENAI_API_KEY'],
    openai_api_base=os.environ['OPENAI_API_BASE'],
    max_tokens=2048,
    temperature=0,
)