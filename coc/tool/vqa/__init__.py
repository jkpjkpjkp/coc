
from .glm import GLM
from .qwen import Qwen25VL
from .gemini import Gemini
glm = GLM()
qwen = Qwen25VL()
gemini = Gemini()

def get_glm():
    return glm._run

def get_qwen():
    return qwen._run

def get_gemini():
    return gemini._run

gemini_as_llm = gemini.invoke