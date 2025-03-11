from coc.tool.sam.mod import SAMWrapper
import gradio as gr
class ExtendSamWrapper(SAMWrapper):
    def create_gradio_interface(self) -> gr.TabbedInterface:
        """Create Gradio interface for interactive use"""
        return gr.TabbedInterface(
            [self._create_generation_interface(), self._create_postprocess_interface()],
            tab_names=["Mask Generation", "Post Processing"],
            title="SAM2 Segmentation Suite"
        )

    def _create_generation_interface(self) -> gr.Interface:
        """Create mask generation interface component"""
        return gr.Interface(
            fn=self._gradio_generate,
            inputs=[
                gr.Image(type="numpy", label="Input Image"),
                *self._create_parameter_controls()
            ],
            outputs=[
                gr.JSON(label="Mask Data"),
                gr.Image(label="Visualization")
            ],
            title="Image Segmentation Generator",
            allow_flagging="never"
        )

    def _create_postprocess_interface(self) -> gr.Interface:
        """Create post-processing interface component"""
        return gr.Interface(
            fn=self._gradio_postprocess,
            inputs=[
                gr.JSON(label="Mask Data"),
                gr.Radio(["binary_mask", "uncompressed_rle", "coco_rle"],
                        value="binary_mask", label="Output Format")
            ],
            outputs=[
                gr.JSON(label="Processed Annotations"),
                gr.Image(label="Visualization")
            ],
            title="Mask Post-Processor",
            allow_flagging="never"
        )

    def _create_parameter_controls(self) -> List[gr.components.Component]:
        """Create standardized parameter controls for Gradio"""
        return [
            gr.Slider(1, 100, step=1, value=32, label="Points per Side"),
            gr.Slider(1, 256, step=1, value=64, label="Points per Batch"),
            gr.Slider(0.0, 1.0, step=0.01, value=0.8, label="IoU Threshold"),
            gr.Slider(0.0, 1.0, step=0.01, value=0.95, label="Stability Threshold"),
            gr.Slider(0.0, 2.0, step=0.01, value=1.0, label="Stability Offset"),
            gr.Slider(-1.0, 1.0, step=0.01, value=0.0, label="Mask Threshold"),
        ]

    def _gradio_generate(self, image: np.ndarray, *args) -> tuple:
        """Wrapper for Gradio generation interface"""
        params = dict(zip([
            'points_per_side',
            'points_per_batch',
            'pred_iou_thresh',
            'stability_score_thresh',
            'stability_score_offset',
            'mask_threshold'
        ], args))

        mask_data = self.generate_masks(image, **params)
        visualization = self._visualize_masks(mask_data)
        return mask_data, visualization

    def _gradio_postprocess(self, mask_data: Dict, output_mode: str) -> tuple:
        """Wrapper for Gradio post-processing interface"""
        annotations = self.post_process_masks(mask_data, output_mode=output_mode)
        visualization = self._visualize_annotations(annotations)
        return annotations, visualization

if __name__ == '__main__':

    sam_engine = SAMWrapper(
        variant='t',
        max_parallel=2  # Adjust based on available GPU memory
    )
    # Create and launch Gradio interface
    interface = sam_engine.create_gradio_interface()
    interface.launch(server_port=sam_port)