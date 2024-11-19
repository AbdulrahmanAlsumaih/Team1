import hydra
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#from PIL import Image
#from torchvision import transforms
from transformers import ViTModel, ViTFeatureExtractor, ViTMSNModel
from scene.gaussian_predictor import GaussianSplatPredictor
from omegaconf import DictConfig, OmegaConf
from myutil import z123Dataset

class Fusion_GaussianSplatPredictor(nn.Module):
    def __init__(self, g_model, cfg, num_heads=8):
        super(Fusion_GaussianSplatPredictor, self).__init__()
        #self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit = ViTMSNModel.from_pretrained("facebook/vit-msn-small")
        # Freeze ViT-MSN weights
        for param in self.vit.parameters():
            param.requires_grad = False
        embed_dim = self.vit.config.hidden_size
        #self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 128 * 128 * 3),  # Map to 3 * 128 * 128
            nn.ReLU(),  # Activation function
            nn.Linear(128 * 128 * 3, 128 * 128 * 3)  # Ensure the output size matches
        )
        self.g_model = g_model
        self.batch = cfg.opt.batch_size

    def forward(self, images,
                source_cameras_view_to_world,
                source_cv2wT_quat=None,
                focals_pixels=None,
                activate_output=True):
        #B = self.batch
        #num_images, C, H, W = 5, 3, 128, 128
        B, num_images, C, H, W = images.shape
        # Resize and normalize images using the feature extractor's parameters
        processed_images = []
        #print("images shape: ", images.shape)
        for i in range(num_images):
            img_batch = images[:, i, :, :, :]  # Shape: (B, C, H, W)
            img_batch_resized = nn.functional.interpolate(img_batch, size=(224, 224))  # Resize to ViT's input size
            processed_images.append(img_batch_resized)

        processed_images = torch.stack(processed_images, dim=1)  # Shape: (B, 5, 3, 224, 224)
        processed_images = processed_images.view(-1, C, 224, 224)  # Flatten to (B * 5, 3, 224, 224)

        # Pass the images through the ViT model
        #self.vit.eval()
        #with torch.no_grad():
        vit_outputs = self.vit(processed_images)  # Get embeddings from the last hidden state
        vit_embeddings = vit_outputs.last_hidden_state  # Shape: (B * 5, num_patches, hidden_dim)

        # Reshape embeddings to group by the number of images per batch
        num_patches, hidden_dim = vit_embeddings.shape[1], vit_embeddings.shape[2]
        vit_embeddings = vit_embeddings.view(B, num_images, num_patches, hidden_dim)  # Shape: (B, 5, num_patches, hidden_dim)

        # Combine the outputs for cross-attention
        vit_outputs_combined = vit_embeddings.view(B, num_images * num_patches, hidden_dim)  # Shape: (B, 5 * num_patches, hidden_dim)

        # Perform cross-attention
        attn_output, attn_weights = self.cross_attention(vit_outputs_combined, vit_outputs_combined, vit_outputs_combined)

        # Apply the MLP to the cross-attention output
        attn_output_flat = attn_output.mean(dim=1)  # Aggregate along the sequence dimension (e.g., average)
        mlp_output = self.mlp(attn_output_flat)  # Shape: (B, 3 * 128 * 128)

        # Reshape to (B, 1, 3, 128, 128)
        output_image = mlp_output.view(B, 1, 3, 128, 128)
        
        output = self.g_model(output_image, source_cameras_view_to_world, source_cv2wT_quat, focals_pixels)

        return output

@hydra.main(version_base=None, config_path='configs', config_name="default_config")
def main(cfg: DictConfig):

    print("start")

    dataset = z123Dataset("/home/cap6411.student1/CVsystem/final_project/splatter-image/my_mini_dataset/train_split", cfg)

    dataloader = DataLoader(dataset,
                            batch_size=cfg.opt.batch_size,
                            shuffle=True,
                            num_workers=0,
                            persistent_workers=False)

    #B, num_images, C, H, W = 2, 5, 3, 128, 128
    #images = torch.randn(B, num_images, C, H, W)

    gaussian_predictor = GaussianSplatPredictor(cfg)
    model = Fusion_GaussianSplatPredictor(gaussian_predictor, cfg)

    focals_pixels_pred = None
    for data in dataloader:
        img = model(data["gt_images"],
                    data["view_to_world_transforms"][:, :cfg.data.input_images, ...],
                    data["source_cv2wT_quat"][:, :cfg.data.input_images],
                    focals_pixels_pred)
        break
    print("end")
    #print(img)

if __name__ == "__main__":
    main()

                                                                                                                                                                                                                                                                                                                                                                           
