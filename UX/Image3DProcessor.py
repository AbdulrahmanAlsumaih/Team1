import sys
import os
os.system('bash setup.sh')
sys.path.append('/home/user/app/splatter-image')
sys.path.append('/home/user/app/diff-gaussian-rasterization')
import torch
import torchvision
import numpy as np
import imageio
from PIL import Image
import rembg
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from io import BytesIO
from utils.app_utils import (
    remove_background,
    resize_foreground,
    set_white_background,
    resize_to_128,
    to_tensor,
    get_source_camera_v2w_rmo_and_quats,
    get_target_cameras,
    export_to_obj
)
from scene.gaussian_predictor import GaussianSplatPredictor
from gaussian_renderer import render_predicted

class Image3DProcessor:
    def __init__(self, model_cfg_path, model_repo_id, model_filename):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        print("Image3DProcessor Device: ", self.device)
        # Load model configuration
        self.model_cfg = OmegaConf.load(model_cfg_path)
        
        # Load pre-trained model weights
        model_path = hf_hub_download(repo_id=model_repo_id, filename=model_filename)
        self.model = GaussianSplatPredictor(self.model_cfg)
        ckpt_loaded = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt_loaded["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def preprocess(self, input_image, preprocess_background=True, foreground_ratio=0.65):
        # Create a new Rembg session
        rembg_session = rembg.new_session()

        # Convert bytes to a PIL image if necessary
        if isinstance(input_image, bytes):
            input_image = Image.open(BytesIO(input_image))

        # Preprocess input image
        if preprocess_background:
            image = input_image.convert("RGB")
            image = remove_background(image, rembg_session)
            image = resize_foreground(image, foreground_ratio)
            image = set_white_background(image)
        else:
            image = input_image
            if image.mode == "RGBA":
                image = set_white_background(image)

        image = resize_to_128(image)

        return image
    @torch.no_grad()
    def reconstruct_and_export(self, image):
        """
        Passes image through model and outputs the reconstruction.
        """
        image= np.array(image)
        image_tensor = to_tensor(image).to(self.device)
        view_to_world_source, rot_transform_quats = get_source_camera_v2w_rmo_and_quats()
        view_to_world_source = view_to_world_source.to(self.device)
        rot_transform_quats = rot_transform_quats.to(self.device)

        reconstruction_unactivated = self.model(
            image_tensor.unsqueeze(0).unsqueeze(0),
            view_to_world_source,
            rot_transform_quats,
            None,
            activate_output=False
        )

        reconstruction = {k: v[0].contiguous() for k, v in reconstruction_unactivated.items()}
        reconstruction["scaling"] = self.model.scaling_activation(reconstruction["scaling"])
        reconstruction["opacity"] = self.model.opacity_activation(reconstruction["opacity"])

        # Render images in a loop
        world_view_transforms, full_proj_transforms, camera_centers = get_target_cameras()
        background = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
        loop_renders = []
        t_to_512 = torchvision.transforms.Resize(512, interpolation=torchvision.transforms.InterpolationMode.NEAREST)

        for r_idx in range(world_view_transforms.shape[0]):
            rendered_image = render_predicted(
                reconstruction,
                world_view_transforms[r_idx].to(self.device),
                full_proj_transforms[r_idx].to(self.device),
                camera_centers[r_idx].to(self.device),
                background,
                self.model_cfg,
                focals_pixels=None
            )["render"]
            rendered_image = t_to_512(rendered_image)
            loop_renders.append(torch.clamp(rendered_image * 255, 0.0, 255.0).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    
        # Save video to a file and load its content
        video_path = "loop_.mp4"
        imageio.mimsave(video_path, loop_renders, fps=25)
        with open(video_path, "rb") as video_file:
            video_data = video_file.read()

        return video_data