temperature = 0

def get_vqa():
    from .glm import GLM
    return GLM()._run