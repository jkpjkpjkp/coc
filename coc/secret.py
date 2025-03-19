import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'

os.environ['ANTHROPIC_API_KEY'] = 'sk-hNEOIT0d5C88QdjDTQPRUf5qwr6sZ8uik6do3t2liehumr49'
os.environ['ANTHROPIC_BASE_URL'] = 'https://chat.cloudapi.vip/v1'

os.environ['DEEPSEEK_API_KEY'] = 'sk-1f5e4b715d244fdaa68952be343ba4bd'
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

os.environ['GEMINI_API_KEY'] = 'sk-5NSZg1YG59VMxq6VUxvL6nrqFSk9RbSyoDo9RmOAE8QkumFU'
os.environ['GEMINI_BASE_URL'] = 'https://api.aigogo.top/v1'

os.environ['GEMINI_API_KEY'] = 'sk-RCVzYMPoI88wxNspjIv3565VesujOuhrcDOqEuy9xi6vcLaJ' # override
os.environ['GEMINI_BASE_URL'] = 'http://www.axs188.club/v1' # override

os.environ['DASHSCOPE_API_KEY'] = 'sk-3a88a7f6a9264d54b62eeb4181192248'

GEMINI_BROKERS = [
    ["sk-RCVzYMPoI88wxNspjIv3565VesujOuhrcDOqEuy9xi6vcLaJ", "http://www.axs188.club/v1"],
    ["sk-5NSZg1YG59VMxq6VUxvL6nrqFSk9RbSyoDo9RmOAE8QkumFU", "https://api.aigogo.top/v1"],
    # Add more brokers from secret.py or configuration
]