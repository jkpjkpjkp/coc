
"""gradio server wrapping coc.tool.grounding.dino. 

NOT TESTED. 
"""
import gradio as gr

def launch():
    with gr.Blocks(title="Grounding DINO Object Detection") as demo:
        gr.Markdown("# Grounding DINO Object Detection")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Input Image")
                objects_input = gr.Textbox(label="Objects to Detect (comma-separated)")
                box_threshold = gr.Slider(0.1, 1.0, value=0.2, step=0.05, label="Box Threshold")
                text_threshold = gr.Slider(0.1, 1.0, value=0.1, step=0.05, label="Text Threshold")
                detect_button = gr.Button("Detect Objects")
            with gr.Column():
                output_image = gr.Image(label="Detection Results")
                output_text = gr.Textbox(label="Detection Details", lines=10)
                output_detections = gr.JSON(label="Detections (for API)", visible=False)  # Added hidden JSON component
        detect_button.click(
            process_dino,
            [image_input, objects_input, box_threshold, text_threshold],
            [output_image, output_text, output_detections],  # Updated outputs
            api_name="predict"
        )
    demo.launch(server_port=dino_port)

def get_dino():
    """launch() if not already, and return a callable as wrapper of api call. """
    pass

if __name__ == "__main__":
    launch()