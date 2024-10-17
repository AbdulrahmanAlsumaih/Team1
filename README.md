````markdown
# Generative Models with Stable Diffusion XL

This project utilizes the Stable Diffusion XL model to generate images based on text prompts and subsequently removes backgrounds using an image segmentation model.

## Setup

### Requirements

Ensure you have the following installed:

- Python 3.8 or higher
- pip
- Access to a GPU (CUDA) is recommended for faster processing

### Installation

1. **Clone the repository** (optional, if you need the code from the repository):
   ```bash
   git clone https://github.com/Stability-AI/generative-models.git
   cd generative-models
   ```
````

2. **Set up a virtual environment**:

   ```bash
   python3 -m venv .pt2
   source .pt2/bin/activate
   ```

3. **Install required packages**:

   ```bash
   pip install -r requirements/pt2.txt
   pip install .
   pip install -e git+https://github.com/Stability-AI/datapipelines.git@main#egg=sdata
   pip install hatch
   ```

4. **Build the project**:

   ```bash
   hatch build -t wheel
   ```

5. **Install additional dependencies**:
   ```bash
   pip install diffusers
   pip install -U transformers
   pip install torch-fidelity
   pip install -qr https://huggingface.co/briaai/RMBG-1.4/resolve/main/requirements.txt
   ```

## Usage

1. **Import the necessary libraries**:

   ```python
   import torch
   from transformers import UMT5EncoderModel
   from diffusers import DiffusionPipeline
   from PIL import Image
   import matplotlib.pyplot as plt
   import os
   import shutil
   ```

2. **Setup device and directories**:

   ```python
   use_cuda = torch.cuda.is_available()
   device = torch.device("cuda" if use_cuda else "cpu")

   if os.path.exists('temp2'):
       shutil.rmtree('temp2')
   os.makedirs("temp2", exist_ok=True)
   ```

3. **Define prompts and generate images**:

   ```python
   prompts = [
       "A simple full body image of an elephant standing on a sidewalk from the side.",
       "A rubber duck sitting on a bare table."
   ]
   images = []

   pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
   pipe.to("cuda")

   for prompt in prompts:
       gen_image = pipe(prompt=prompt).images[0]
       gen_image.save(f"temp2/{prompt.replace(' ', '_')}.jpg")
       images.append(gen_image)
   ```

4. **Remove backgrounds**:

   ```python
   from transformers import pipeline

   pipe2 = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device=device)

   for i, image in enumerate(images):
       final_image = pipe2(image)  # applies mask on input and returns a pillow image
       final_image.convert('RGB').save(f"temp2/NoBG-image-{i}.jpg")
   ```

## Results

Generated images and their background-removed versions will be saved in the `temp2` directory.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

```

Feel free to copy and use it!
```
