import hydra
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#from .dataset_readers import readCamerasFromTxt
from utils.general_utils import PILtoTorch, matrix_to_quaternion
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getView2World

from PIL import Image
from omegaconf import DictConfig, OmegaConf
from compute_c2w import camera_to_world_matrix

class z123Dataset(Dataset):
    def __init__(self, img_dir, cfg, transform=None):
        self.cfg = cfg
        self.img_dir = img_dir
        self.img_folders = sorted(os.listdir(self.img_dir))
        self.angles = [(30, 20), (90, -10), (150, 20), (210, -10), (270, 20), (330, -10)]
        self.transform = transform

        self.imgs_per_obj = self.cfg.opt.imgs_per_obj

        self.projection_matrix = getProjectionMatrix(
            znear=self.cfg.data.znear, zfar=self.cfg.data.zfar,
            fovX=30 * 2 * np.pi / 360, 
            fovY=30 * 2 * np.pi / 360).transpose(0,1)
        
        self.cam_infos = []

        for a,e in self.angles:
            R, T = camera_to_world_matrix(a, e)
            pair_dict = {}
            pair_dict["R"] = R
            pair_dict["T"] = T
            self.cam_infos.append(pair_dict)

    def __len__(self):
        return len(self.img_folders)

    def make_poses_relative_to_first(self, images_and_camera_poses):
        inverse_first_camera = images_and_camera_poses["world_view_transforms"][0].inverse().clone()
        for c in range(images_and_camera_poses["world_view_transforms"].shape[0]):
            images_and_camera_poses["world_view_transforms"][c] = torch.bmm(
                                                inverse_first_camera.unsqueeze(0),
                                                images_and_camera_poses["world_view_transforms"][c].unsqueeze(0)).squeeze(0)
            images_and_camera_poses["view_to_world_transforms"][c] = torch.bmm(
                                                images_and_camera_poses["view_to_world_transforms"][c].unsqueeze(0),
                                                inverse_first_camera.inverse().unsqueeze(0)).squeeze(0)
            images_and_camera_poses["full_proj_transforms"][c] = torch.bmm(
                                                inverse_first_camera.unsqueeze(0),
                                                images_and_camera_poses["full_proj_transforms"][c].unsqueeze(0)).squeeze(0)
            images_and_camera_poses["camera_centers"][c] = images_and_camera_poses["world_view_transforms"][c].inverse()[3, :3]
        return images_and_camera_poses
    
    def get_source_cw2wT(self, source_cameras_view_to_world):
        # Compute view to world transforms in quaternion representation.
        # Used for transforming predicted rotations
        qs = []
        for c_idx in range(source_cameras_view_to_world.shape[0]):
            qs.append(matrix_to_quaternion(source_cameras_view_to_world[c_idx, :3, :3].transpose(0, 1)))
        return torch.stack(qs, dim=0)

    def load_example_id(self, example_id, dir_path,
                        trans = np.array([0.0, 0.0, 0.0]), scale=1.0):
        
        #rgb_paths = sorted(glob.glob(os.path.join(dir_path, example_id, 'cropped', "*")))

        if not hasattr(self, "all_rgbs"):
            self.all_rgbs = {}
            self.all_world_view_transforms = {}
            self.all_view_to_world_transforms = {}
            self.all_full_proj_transforms = {}
            self.all_camera_centers = {}

        if example_id not in self.all_rgbs.keys():
            self.all_rgbs[example_id] = []
            self.all_world_view_transforms[example_id] = []
            self.all_full_proj_transforms[example_id] = []
            self.all_camera_centers[example_id] = []
            self.all_view_to_world_transforms[example_id] = []

            for i in range(6):
                R = self.cam_infos[i]["R"]
                T = self.cam_infos[i]["T"]
                image_path = os.path.join(dir_path, example_id, "cropped", f"cropped_image{i}.png")
                image = Image.open(image_path)
                self.all_rgbs[example_id].append(PILtoTorch(image, 
                                                            (self.cfg.data.training_resolution, self.cfg.data.training_resolution)).clamp(0.0, 1.0)[:3, :, :])

                world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
                view_world_transform = torch.tensor(getView2World(R, T, trans, scale)).transpose(0, 1)

                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]

                self.all_world_view_transforms[example_id].append(world_view_transform)
                self.all_view_to_world_transforms[example_id].append(view_world_transform)
                self.all_full_proj_transforms[example_id].append(full_proj_transform)
                self.all_camera_centers[example_id].append(camera_center)
            
            self.all_world_view_transforms[example_id] = torch.stack(self.all_world_view_transforms[example_id])
            self.all_view_to_world_transforms[example_id] = torch.stack(self.all_view_to_world_transforms[example_id])
            self.all_full_proj_transforms[example_id] = torch.stack(self.all_full_proj_transforms[example_id])
            self.all_camera_centers[example_id] = torch.stack(self.all_camera_centers[example_id])
            self.all_rgbs[example_id] = torch.stack(self.all_rgbs[example_id])

    def get_example_id(self, index):
        example_id = self.img_folders[index]
        
        return example_id

    def __getitem__(self, idx):
        example_id = self.img_folders[idx]

        self.load_example_id(example_id, self.img_dir)

        frame_idxs = torch.randperm(
                    len(self.all_rgbs[example_id])
                    )[:self.imgs_per_obj]

        frame_idxs = torch.cat([frame_idxs[:self.cfg.data.input_images], frame_idxs], dim=0)

        images_and_camera_poses = {
            "gt_images": self.all_rgbs[example_id][frame_idxs].clone(),
            "world_view_transforms": self.all_world_view_transforms[example_id][frame_idxs],
            "view_to_world_transforms": self.all_view_to_world_transforms[example_id][frame_idxs],
            "full_proj_transforms": self.all_full_proj_transforms[example_id][frame_idxs],
            "camera_centers": self.all_camera_centers[example_id][frame_idxs]
        }
        images_and_camera_poses = self.make_poses_relative_to_first(images_and_camera_poses)
        images_and_camera_poses["source_cv2wT_quat"] = self.get_source_cw2wT(images_and_camera_poses["view_to_world_transforms"])

        return images_and_camera_poses

@hydra.main(version_base=None, config_path='configs', config_name="default_config")
def main(cfg: DictConfig):

    dataset = z123Dataset("/home/cap6411.student1/CVsystem/final_project/splatter-image/my_mini_dataset/train_split", cfg)

    dataloader = DataLoader(dataset, 
                            batch_size=cfg.opt.batch_size,
                            shuffle=True,
                            num_workers=0,
                            persistent_workers=False)

    for data in dataloader:
        #print("data: ", data)
        input_images = data["gt_images"][:, :1, ...]
        print("gt", data["gt_images"].shape)
        print(input_images.shape)


if __name__ == "__main__":

    main()




