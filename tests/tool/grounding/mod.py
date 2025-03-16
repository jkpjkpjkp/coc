from coc.tool.grounding.mod import ObjectDetectionFactory, get_grounding
from coc.tool.grounding.dino import draw_boxes
import PIL.Image

def demo1():
    obj = ObjectDetectionFactory()
    image_path = '/home/jkp/hack/coc/data/sample/onions.jpg'
    image = PIL.Image.open(image_path)
    ret = obj._run(texts=['boy', 'girl', 'an onion'], image=image)
    if isinstance(ret, tuple) and len(ret) == 1:
        ret = ret[0]
    print(ret)
    res = draw_boxes(image, ret)
    res.save('raw_onions.jpg')

def demo2():
    import gradio as gr
    demo = gr.Interface(
        fn=get_grounding(),
        inputs=[
            gr.Image(type="pil"),
            gr.Textbox(label="Objects of Interest (comma-separated)")
        ],
        outputs=[
            gr.JSON(label="Raw Detection Data"),
        ],
        title="Object Grounding Demo",
        description="Upload an image and specify objects to detect, separated by commas."
    )
    app = demo.launch(share=False)

if __name__ == '__main__':
    demo2()