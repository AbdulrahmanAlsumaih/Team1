import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Step 1: Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # Shape: (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2)  # Shape: (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # Shape: (B, num_patches, embed_dim)
        return x

# Step 2: Cross-Attention Mechanism
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output

# Step 3: Transformer Block with Cross-Attention
class TransformerBlockWithCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlockWithCrossAttention, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dim_reduction = nn.Linear(embed_dim, 128)
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)    

    def forward(self, img1_tokens, img2_tokens):
        attn_output, _ = self.cross_attn(img1_tokens, img2_tokens, img2_tokens)
        img1_tokens = self.norm1(attn_output + img1_tokens)
        
        ff_output = self.ff(img1_tokens)
        img1_tokens = self.norm2(ff_output + img1_tokens)
        
        # Reshape from (B, 196, 128) to (B, 128, 14, 14)
        reduced_features = self.dim_reduction(img1_tokens)
        num_patches = int(reduced_features.size(1) ** 0.5)  # Assuming a 14x14 grid
        reshaped_features = reduced_features.view(-1, 128, num_patches, num_patches)  # Shape: (B, 128, 14, 14)
                
        #Upsample to (B, 128, 224, 224)
        upsampled_features = self.upsample(reshaped_features)  # Shape: (B, 128, 224, 224)

        return upsampled_features

if __name__ == "__main__":
    img_size = 224
    patch_size = 16
    embed_dim = 768
    num_heads = 8
    ff_dim = 2048

    transform = transforms.Compose([
        transforms.Resize(256),            # Resize the image to 256x256 (or similar size)
        transforms.CenterCrop(224),        # Crop the center of the image to get 224x224 size
        transforms.ToTensor(),             # Convert image to PyTorch tensor (C, H, W) and scale to [0, 1]
        transforms.Normalize(              # Normalize the image with mean and std for each channel (ImageNet mean/std)
            mean=[0.485, 0.456, 0.406],    # Mean for R, G, B channels (ImageNet)
            std=[0.229, 0.224, 0.225]      # Standard deviation for R, G, B channels (ImageNet)
        ),
    ]) 


    # Initialize patch embedding and transformer block
    patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
    cross_attn_block = TransformerBlockWithCrossAttention(embed_dim, num_heads, ff_dim)

    img_path1 = "./test_img/front_XL-image-NoBG0.jpg"
    img_path2 = "./test_img/back_XL-image-NoBG0.jpg"

    image1 = Image.open(img_path1).convert('RGB')
    image2 = Image.open(img_path2).convert('RGB')

    image_tensor1 = transform(image1)  # Shape: (3, 224, 224)
    image_tensor2 = transform(image2)  # Shape: (3, 224, 224)

    img1 = image_tensor1.unsqueeze(0)
    img2 = image_tensor2.unsqueeze(0)

    # Example images: img1 and img2 (assume preprocessed tensors of shape (B, 3, 224, 224))
    img1_patches = patch_embed(img1)
    img2_patches = patch_embed(img2)

    # Perform cross-attention
    cross_attn_output = cross_attn_block(img1_patches, img2_patches)

    print("=================")
    print(cross_attn_output)
    print("size: ", cross_attn_output.size())






