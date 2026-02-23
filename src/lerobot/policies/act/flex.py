import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexVLA(nn.Module):
    """
    Flex VLA model integrated with DINOv2 backbone.
    """
    def __init__(
        self,
        num_scene_tokens=900,      
        num_cameras=2,             
        num_timesteps=1,           
        target_tokens_per_img=160, 
        d_llm=512,                 
        encoder_layers=8,
        patch_size=14
    ):
        super().__init__()
        
        # 1. Patchifier: Load DINOv2-Base (ViT-B/14)
        try:
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self.backbone.head = nn.Identity() 
            self.use_hf = False
        except Exception:
            print("Torch Hub failed, falling back to HuggingFace Transformers...")
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")
            self.use_hf = True

        embed_dim = 768  
        
        self.num_cameras = num_cameras
        self.num_timesteps = num_timesteps
        self.num_scene_tokens = num_scene_tokens
        self.target_tokens_per_img = target_tokens_per_img
        self.patch_size = patch_size

        # 2. Scene tokens and positional embeddings
        self.scene_tokens = nn.Parameter(torch.randn(1, num_scene_tokens, embed_dim))
        self.cam_embed = nn.Parameter(torch.randn(1, num_cameras, 1, embed_dim))
        self.time_embed = nn.Parameter(torch.randn(1, 1, num_timesteps, embed_dim))

        # 3. Flex Scene Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=12, dim_feedforward=3072, 
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.flex_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)

        # 4. Projection to LLM dimension
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, d_llm)
        )

    def forward(self, x):
        """
        Args:
            x: Batch of images. Shape (B, C, T, 3, H, W) or (B, C, 3, H, W).
        """
        if x.dim() == 5: # (B, C, 3, H, W)
             x = x.unsqueeze(2) # (B, C, 1, 3, H, W)

        B, C, T, _, H, W = x.shape
        
        # Flatten for DINOv2
        x = x.view(B * C * T, 3, H, W)
        
        # DINOv2 intermediate features
        if self.use_hf:
            # HuggingFace DINOv2 output
            outputs = self.backbone(pixel_values=x)
            # last_hidden_state: (Batch, Seq, Dim). Seq = N + 1 (CLS)
            features = outputs.last_hidden_state[:, 1:, :] 
        else:
            # TorchHub DINOv2 output
            features = self.backbone.get_intermediate_layers(x, n=1)[0] # (B*C*T, N, D)
        
        D = features.shape[-1]
        
        # Calculate grid size
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        
        # Reshape to grid
        # If N != grid_h*grid_w (e.g. padding), we might have issues. 
        # For now assume standard usage where H, W are multiples of 14.
        tokens_grid = features.transpose(1, 2).reshape(B * C * T, D, grid_h, grid_w)
        
        # Downsample to target tokens
        # We approximate target grid size to match target_tokens_per_img
        # For 160 tokens, 10x16 is good.
        # Let's derive target_h/w from aspect ratio.
        target_area = self.target_tokens_per_img
        ratio = H / W
        target_w = int((target_area / ratio)**0.5)
        target_h = int(target_area / target_w)
        
        # Fix for small rounding errors
        if target_h * target_w != target_area and target_area == 160:
             target_h, target_w = 10, 16 

        downsampled = F.interpolate(tokens_grid, size=(target_h, target_w), mode='bilinear', align_corners=False)
        img_tokens = downsampled.flatten(2).transpose(1, 2) # (B*C*T, target_tokens, D)
        
        # Restore grouping
        img_tokens = img_tokens.view(B, C, T, -1, D)

        # Add positional embeddings
        # Expand/Slice embeddings to match input
        # This allows training with T=1 or T=9, etc.
        cam_emb = self.cam_embed[:, :C, :, :]
        time_emb = self.time_embed[:, :, :T, :]
        
        img_tokens = img_tokens + cam_emb.unsqueeze(3) + time_emb.unsqueeze(3)
        img_tokens = img_tokens.reshape(B, -1, D) 

        # Concat with scene tokens
        scene_tokens = self.scene_tokens.expand(B, -1, -1)
        combined = torch.cat([scene_tokens, img_tokens], dim=1)
        
        # Flex Encoder
        updated = self.flex_encoder(combined)
        
        # Extract scene tokens
        flex_rep = updated[:, :self.num_scene_tokens, :]
        
        # Project
        return self.proj(flex_rep)