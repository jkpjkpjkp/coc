"""

UNTESTED. UNFINISHED.
refer to demo() below.
"""

def demo():
    import requests
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForCausalLM

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

    prompt = "<OD>"

    image = Image.open('data/sample/4girls.jpg')

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))

    print(parsed_answer)

import gradio as gr
from PIL import Image
import requests
# Load model directly
from transformers import AutoModelForCausalLM, AutoProcessor
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("Microsoft/Florence-2-base", trust_remote_code=True)

def run_example(task_prompt, text_input=None, image=None):
    if image is None:
        image = Image.open(requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true", stream=True).raw);
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device, torch_dtype); generated_ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, do_sample=False, num_beams=3); generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]; if task_prompt == "<OD>": result = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height)); else: result = generated_text; return result

def vlm(*args):
    """arbitrary string and image input.

    Process a variable number of string and PIL image arguments into a prompt and image list,
    maintaining their relative positions, and generate a response using the pipeline.

    Args:
        *args: Variable arguments, each being either a str or a PIL.Image.

    Returns:
        str: Generated text from the pipeline.
    """
    prompt = ""
    images = []

    # Build prompt and collect images while preserving order
    for arg in args:
        if isinstance(arg, str):
            prompt += arg
        elif isinstance(arg, Image.Image):
            prompt += " <image>"
            images.append(arg)
        else:
            raise ValueError("Arguments must be str or PIL.Image.Image")

    result = model.generate(prompt) if not images else model.generate(prompt, images=images)

    # Extract and return the generated text
    # Note: The exact output structure might depend on the pipeline's return format
    return result[0]['generated_text'] if isinstance(result, list) else result

if __name__ == '__main__':
    print(vlm('hi! what is your name? '))