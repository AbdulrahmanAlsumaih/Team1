import torch
from transformers import UMT5EncoderModel
from diffusers import AutoPipelineForText2Image
from diffusers import DiffusionPipeline
import torch
import pandas as pd
import time
import os
import shutil
import torchvision
from PIL import Image
from transformers import pipeline
import matplotlib.pyplot as plt

def sdxl_bg_remove(prompts):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    # Set proper device based on cuda availability
    device = torch.device("cuda" if use_cuda else "cpu")

    #batch_size = 10


    if os.path.exists('temp2'):
        shutil.rmtree('temp2')
    os.makedirs("temp2", exist_ok=True)

    paths = ["temp1", "temp2"]
    
    images=[]

    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")

    start_time = time.time()
    for i, prompt in enumerate(prompts):
        gen_image = pipe(prompt=prompt).images[0]
        #gen_image.save(f"XL-image{i}.jpg")
        #plt.imshow(gen_image)
        #plt.show()
        images.append(gen_image)

    end_time = time.time()
    print("Total Time SDXL: %4f seconds" % (end_time - start_time))

    start_time = time.time()
    for i, image in enumerate(images):
        #image_resized = image.resize((376, 263), Image.Resampling.LANCZOS)
        #image_resized.save(f"temp2/XL-image{i}.jpg")

        pipe2 = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device=device)
        #pillow_mask = pipe2(image, return_mask = True) # outputs a pillow mask
        final_image = pipe2(image) # applies mask on input and returns a pillow image
        final_image.convert('RGB').save(f"XL-image-NoBG{i}.jpg")
        #plt.imshow(final_image)
        #plt.show()
    
    end_time = time.time()
    print("Total Time BG Removal: %4f seconds" % (end_time - start_time))
    return final_image

if __name__ == '__main__':

    prompts = ["a crab, low poly",
    "a bald eagle carved out of wood",
    "a delicious hamburger"]

    result = sdxl_bg_remove(prompts)
    




