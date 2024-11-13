import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import ViTModel, ViTFeatureExtractor

class CrossAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        
        return attn_output

def reshape_feature(feature):
    cross_attention_output = feature[:, 1:, :]
    # Reshape (B, num_patches, embed_dim) -> (B, 14, 14, embed_dim)
    grid_size = int(cross_attention_output.shape[1] ** 0.5)  # E.g., 14 for 196 patches
    cross_attention_output = cross_attention_output.view(-1, grid_size, grid_size, cross_attention_output.size(-1))

    # Permute to match U-Net's input (B, embed_dim, H, W)
    cross_attention_output = cross_attention_output.permute(0, 3, 1, 2)  # Shape: (B, embed_dim, 14, 14) 
    
    upsample = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)
    upsampled_output = upsample(cross_attention_output)  # Shape: (B, 768, 128, 128)
    
    conv = nn.Conv2d(in_channels=768, out_channels=3, kernel_size=1)
    final_output = conv(upsampled_output)  # Shape: (B, 3, 128, 128)

    return final_output

if __name__ == "__main__":

    # Load pretrained ViT model from Hugging Face
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    img_path1 = "./test_img/front_XL-image-NoBG0.jpg"
    img_path2 = "./test_img/back_XL-image-NoBG0.jpg"

    img1 = Image.open(img_path1)
    img2 = Image.open(img_path2)

    inputs_img1 = feature_extractor(images=img1, return_tensors="pt")
    inputs_img2 = feature_extractor(images=img2, return_tensors="pt")

    # Extract image features (patch embeddings)
    with torch.no_grad():
        img1_features = model(**inputs_img1).last_hidden_state  # Shape: (B, num_patches, embed_dim)
        img2_features = model(**inputs_img2).last_hidden_state

    # Define cross-attention layer
    embed_dim = img1_features.shape[-1]  # 768
    num_heads = 8
    cross_attention = CrossAttention(embed_dim, num_heads)

    # Apply cross-attention (img1 features as query, img2 features as key/value)
    cross_attention_output = cross_attention(img1_features, img2_features, img2_features)
    
    reshape_att = reshape_feature(cross_attention_output)

    print(reshape_att.shape)  # Shape: (1, num_patches, 768)

    torch.save(reshape_att[0], "fused_tensor.pt")



