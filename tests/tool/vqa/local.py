import gradio as gr
from PIL import Image
from transformers import pipeline
from coc.config import local_vlm_port

pipe = pipeline("image-text-to-text", model="microsoft/Florence-2-base", trust_remote_code=True)

def vlm(*args):
    """
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
            raise ValueError("Arguments must be str or PIL.Image")

    # Handle cases based on presence of images
    if not images:
        # Pure text input
        result = pipe(prompt)
    else:
        # Text with images
        result = pipe(prompt, images=images)

    # Extract and return the generated text
    # Note: The exact output structure might depend on the pipeline's return format
    return result[0]['generated_text'] if isinstance(result, list) else result

# Wrapper function for Gradio with fixed inputs
def gradio_func(text1, image1, text2, image2):
    """
    Wrapper function for Gradio interface, accepting two text-image pairs.

    Args:
        text1 (str): First text input.
        image1 (PIL.Image): First image input.
        text2 (str): Second text input.
        image2 (PIL.Image): Second image input.

    Returns:
        str: Generated text from vlm.
    """
    return vlm(text1, image1, text2, image2)

# Set up the Gradio interface
iface = gr.Interface(
    fn=gradio_func,
    inputs=[
        gr.Textbox(label="Text 1", placeholder="Enter first text segment"),
        gr.Image(type="pil", label="Image 1"),
        gr.Textbox(label="Text 2", placeholder="Enter second text segment"),
        gr.Image(type="pil", label="Image 2")
    ],
    outputs=gr.Textbox(label="Generated Output"),
    title="Vision-Language Model Demo",
    description="Enter text segments and upload images in sequence. The model will process them into a single prompt."
)

# Launch the server
iface.launch(server_port=local_vlm_port)