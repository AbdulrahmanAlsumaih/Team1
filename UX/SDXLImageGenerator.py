import torch
from diffusers import DiffusionPipeline
import time
from PIL import Image
from io import BytesIO

class SDXLImageGenerator:
    def __init__(self):
        # Check if cuda is available
        self.use_cuda = torch.cuda.is_available()
        # Set proper device based on cuda availability
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        print("SDXLImageGenerator Device: ", self.device)

        # Load the pipeline
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        self.pipe.to(self.device)

    def generate_image(self, prompts):
        start_time = time.time()
        
        # Generate images in a batch
        outputs = self.pipe(prompt=prompts)
        images = outputs.images

        # Convert images to PNG byte data
        png_images = []
        for image in images:
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)  # Ensure the buffer is at the start for reading
            png_images.append(buffer.getvalue())  # PNG data in bytes

        end_time = time.time()
        print("Total Time SDXL: %4f seconds" % (end_time - start_time))
        return png_images  # List of PNG byte data