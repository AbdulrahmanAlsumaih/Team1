import os
import gradio as gr
import tempfile
from SDXLImageGenerator import SDXLImageGenerator  # Import your existing class
import sys
from Image3DProcessor import Image3DProcessor  # Import your 3D processing class
from PIL import Image
import io
from io import BytesIO
import numpy as np

class VideoGenerator:
    def __init__(self, model_cfg_path, model_repo_id, model_filename):
        # Initialize the Image3DProcessor
        self.processor = Image3DProcessor(model_cfg_path, model_repo_id, model_filename)

    def generate_3d_video(self, image):
        # Ensure the image is a PIL Image object
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        # Preprocess the image first
        processed_image = self.processor.preprocess(image)
        # Then pass it to reconstruct_and_export
        video_data = self.processor.reconstruct_and_export(processed_image)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as video_file:
            video_file.write(video_data)
            video_path = video_file.name
        return video_path

class GradioApp:
    def __init__(self):
        self.sdxl_generator = SDXLImageGenerator()
        # Initialize VideoGenerator with required paths and details
        self.video_generator = VideoGenerator(
            model_cfg_path="/home/user/app/splatter-image/gradio_config.yaml",
            model_repo_id="szymanowiczs/splatter-image-multi-category-v1",
            model_filename="model_latest.pth"
        )

    def launch(self):
        with gr.Blocks() as interface:
            # Input for the prompt at the top
            prompt_input = gr.Textbox(label="Input Prompt", elem_id="input_textbox")

            # Button for generating the 3D object
            generate_3d_object = gr.Button("Generate 3D object")

            # Outputs: image on the bottom left, video on the bottom right
            with gr.Row():
                with gr.Column():
                    image_output = gr.Image(label="Generated Image", elem_id="generated_image")
                with gr.Column():
                    video_output = gr.Video(label="3D Model Video", elem_id="model_video")

            # Generate the image first
            def generate_image_and_display(prompt):
                # Generate the image from the prompt
                image_data = self.sdxl_generator.generate_image([prompt])[0]
                return Image.open(BytesIO(image_data))

            # Generate the 3D after the image is ready
            def generate_3D_from_image(image):
                # Ensure the image is a PIL Image object
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                # Generate the 3D from the generated image
                return self.video_generator.generate_3d_video(image)

            # First click generates the image
            generate_3d_object.click(
                fn=generate_image_and_display,
                inputs=prompt_input,
                outputs=image_output,
                queue=True
            )

            # Once the image is ready, generate the video
            image_output.change(
                fn=generate_3D_from_image,
                inputs=image_output,
                outputs=video_output,
                queue=True
            )

        interface.launch(share=True)

if __name__ == "__main__":
    app = GradioApp()
    app.launch()
