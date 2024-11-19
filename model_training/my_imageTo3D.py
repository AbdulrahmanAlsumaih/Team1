import torch
import torchvision
import numpy as np
import os
from omegaconf import OmegaConf
from PIL import Image

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

import imageio
import rembg
from huggingface_hub import hf_hub_download
from scene.gaussian_predictor import GaussianSplatPredictor
from scene.gaussian_predictor import GaussianSplatPredictor
from gaussian_renderer import render_predicted

@torch.no_grad()
def preprocess(input_image, preprocess_background=True, foreground_ratio=0.65):
    # Create a new Rembg session
    rembg_session = rembg.new_session()

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


def reconstruct_and_export(image, image_name, out_path="./myoutput"):
    """
    Passes image through model and outputs the reconstruction.
    """
    device = "cuda"
    image = to_tensor(image).to(device)
    view_to_world_source, rot_transform_quats = get_source_camera_v2w_rmo_and_quats()
    view_to_world_source = view_to_world_source.to(device)
    rot_transform_quats = rot_transform_quats.to(device)

    reconstruction_unactivated = model(
        image.unsqueeze(0).unsqueeze(0),
        view_to_world_source,
        rot_transform_quats,
        None,
        activate_output=False
    )


    reconstruction = {k: v[0].contiguous() for k, v in reconstruction_unactivated.items()}
    reconstruction["scaling"] = model.scaling_activation(reconstruction["scaling"])
    reconstruction["opacity"] = model.opacity_activation(reconstruction["opacity"])

    # Render images in a loop
    world_view_transforms, full_proj_transforms, camera_centers = get_target_cameras()
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
    loop_renders = []
    t_to_512 = torchvision.transforms.Resize(512, interpolation=torchvision.transforms.InterpolationMode.NEAREST)

    for r_idx in range(world_view_transforms.shape[0]):
        rendered_image = render_predicted(
            reconstruction,
            world_view_transforms[r_idx].to(device),
            full_proj_transforms[r_idx].to(device),
            camera_centers[r_idx].to(device),
            background,
            model_cfg,
            #focals_pixels=None
        )["render"]
        rendered_image = t_to_512(rendered_image)
        loop_renders.append(torch.clamp(rendered_image * 255, 0.0, 255.0).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8))

    loop_out_path = f"{out_path}/{image_name}.mp4"
    imageio.mimsave(loop_out_path, loop_renders, fps=25)
    ply_out_path = f"{out_path}/{image_name}_mesh.ply"
    export_to_obj(reconstruction_unactivated, ply_out_path)

    return ply_out_path, loop_out_path

def get_splatter_image(images, out_path):

    config_path = "./gradio_config.yaml"
    model_cfg = OmegaConf.load(config_path)

    model_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-multi-category-v1",
                                filename="model_latest.pth")
    model = GaussianSplatPredictor(model_cfg)
    ckpt_loaded = torch.load(model_path, map_location="cuda")
    model.load_state_dict(ckpt_loaded["model_state_dict"])
    model.to("cuda")

    for i in range(len(images)):
        process_image = preprocess(images[i], preprocess_background=False, foreground_ratio=0.65)

        ply_out_path, loop_out_path = reconstruct_and_export(np.array(process_image), f"{i}", out_path)

        print(f"3D model saved to {ply_out_path}")
        print(f"Video render saved to {loop_out_path}")

if __name__ == '__main__':
    # Specify the path directly
    config_path = "./gradio_config.yaml"  # Replace with the actual path
    model_cfg = OmegaConf.load(config_path)


    # Load pre-trained model weights
    model_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-multi-category-v1",
                                filename="model_latest.pth")
    model = GaussianSplatPredictor(model_cfg)
    ckpt_loaded = torch.load(model_path, map_location="cuda")
    model.load_state_dict(ckpt_loaded["model_state_dict"])
    model.to("cuda")

    image = Image.open("./front_XL-image-NoBG0.jpg")

    process_image = preprocess(image, preprocess_background=False, foreground_ratio=0.65)

    # Perform reconstruction and export results
    ply_out_path, loop_out_path = reconstruct_and_export(np.array(process_image), "XL-image-NoBG0")

    print(f"3D model saved to {ply_out_path}")
    print(f"Video render saved to {loop_out_path}")





