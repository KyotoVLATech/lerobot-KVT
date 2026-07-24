from abc import ABC
from functools import partial
from typing import Tuple, Union, Dict

import numpy as np
import timm
import torch
from timm.models.vision_transformer import Block
from torch import nn as nn
from torch.nn import functional as F

# from robobase import utils
# from robobase.models.core import (
#     RoboBaseModule,
#     get_activation_fn_from_str,
#     get_normalization_fn_from_str,
# )
from rl_100.model.vision.utils import (
    MultiViewConvEmbed,
    MultiViewPatchEmbed,
    get_2d_sincos_pos_embed,
)

# Add imports for the new functionality
import copy
import torchvision
from typing import Dict, Union
import einops


class EncoderModule(nn.Module, ABC):
    def __init__(self, input_shape: Tuple[int, int, int, int]):
        super().__init__()
        self.input_shape = input_shape
        assert (
            len(input_shape) == 4
        ), f"Expected shape (V, C, H, W), but got {input_shape}"


class EncoderCNNMultiViewDownsampleWithStrides(EncoderModule):
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        num_downsample_convs: int = 1,
        num_post_downsample_convs: int = 3,
        channels: int = 32,
        kernel_size: int = 3,
        padding: int = 0,
        channels_multiplier: int = 1,
        activation: str = "relu",
        norm: str = "identity",
        normalise_inputs: bool = True,
    ):
        super().__init__(input_shape)
        self._normalise_inputs = normalise_inputs
        num_cameras = input_shape[0]
        self.activation_fn = get_activation_fn_from_str(activation)
        self.norm_fn = get_normalization_fn_from_str(norm)
        self.convs_per_cam = nn.ModuleList()
        final_channels = 0
        for i in range(num_cameras):
            resolution = np.array(input_shape[2:])
            net = []
            input_channels = input_shape[1]
            output_channels = channels
            for _ in range(num_downsample_convs):
                net.append(
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                    )
                )
                net.append(self.norm_fn(output_channels))
                net.append(self.activation_fn())
                input_channels = output_channels
                output_channels *= channels_multiplier
                resolution = np.floor((resolution + 2 * padding - kernel_size) / 2) + 1
            for _ in range(num_post_downsample_convs):
                net.append(
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                    )
                )
                net.append(self.norm_fn(output_channels))
                net.append(self.activation_fn())
                input_channels = output_channels
                output_channels *= channels_multiplier
                resolution = np.floor((resolution + 2 * padding - kernel_size) / 1) + 1
            self.convs_per_cam.append(nn.Sequential(*net))
            final_channels = int(input_channels * resolution.prod())
        self._output_shape = (num_cameras, final_channels)
        self.apply(utils.weight_init)

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x):
        assert (
            self.input_shape == x.shape[1:]
        ), f"expected input shape {self.input_shape} but got {x.shape[1:]}"
        if self._normalise_inputs:
            x = x / 255.0 - 0.5
        outs = []
        for _x, net in zip(x.unbind(1), self.convs_per_cam):
            outs.append(net(_x).view(-1, self.output_shape[-1]))
        fused = torch.stack(outs, 1)
        assert (
            fused.shape[1:] == self.output_shape
        ), f"Expected output {self.output_shape}, but got {fused.shape[1:]}"
        return fused


class EncoderMVPMultiView(EncoderModule):
    _OUT_DIM = {"vitb-mae-egosoup": 768, "vits-mae-hoi": 384}
    # Per-channel mean and standard deviation (in RGB order)
    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def __init__(
        self, input_shape: Tuple[int, int, int, int], name: str = "vitb-mae-egosoup"
    ):
        super().__init__(input_shape)
        assert tuple(input_shape[2:]) == (
            224,
            224,
        ), f"MVP requires images of shape (224, 224), but got {input_shape[2:]}"
        assert input_shape[1] == 3, "MVP only supports channel of size 3"
        # Per-channel mean and standard deviation (in RGB order)
        try:
            import mvp
            import ssl

            ssl._create_default_https_context = ssl._create_unverified_context
        except ImportError:
            raise ImportError(
                "Please run: pip install git+https://github.com/ir413/mvp"
            )
        self._mvp = mvp.load(name)
        self._mvp_out = EncoderMVPMultiView._OUT_DIM[name]
        self._mvp.freeze()

    def _color_norm(self, im, mean, std):
        """Performs per-channel normalization."""
        for i in range(3):
            im[..., i, :, :] = (im[..., i, :, :] - mean[i]) / std[i]
        return im

    def forward(self, x):
        assert (
            self.input_shape == x.shape[1:]
        ), f"expected input shape {self.input_shape} but got {x.shape[1:]}"
        x = self._color_norm(
            x / 255.0, EncoderMVPMultiView._MEAN, EncoderMVPMultiView._STD
        )
        with torch.no_grad():
            outs = []
            for _x in x.unbind(1):
                outs.append(self._mvp(_x).view(-1, self.output_shape[-1]))
            fused = torch.stack(outs, 1)
            assert (
                fused.shape[1:] == self.output_shape
            ), f"Expected output {self.output_shape}, but got {fused.shape[1:]}"
            return fused

    @property
    def output_shape(self):
        return self.input_shape[0], self._mvp_out


class ResNetEncoder(EncoderModule):
    # Per-channel mean and standard deviation (in RGB order)
    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        model: str,
    ):
        super().__init__(input_shape)
        assert input_shape[1] == 3, "ResNet only supports channel of size 3"
        self.model = timm.create_model(model, pretrained=True)
        self.model.eval()

    def forward(self, x: torch.Tensor):
        assert (
            self.input_shape == x.shape[1:]
        ), f"expected input shape {self.input_shape} but got {x.shape[1:]}"
        B, V = x.shape[:2]
        with torch.no_grad():
            outs = []
            for _x in x.unbind(1):
                _x = F.interpolate(
                    _x, size=(224, 224), mode="bilinear", align_corners=False
                )
                _x = self._color_norm(_x / 255.0)
                _x = self.model.forward_features(_x)
                _x = self.model.global_pool(_x)
                outs.append(_x.view(B, -1))
            fused = torch.stack(outs, 1)
        return fused

    def _color_norm(self, im):
        """Performs per-channel normalization."""
        mean = ResNetEncoder._MEAN
        std = ResNetEncoder._STD
        for i in range(3):
            im[..., i, :, :] = (im[..., i, :, :] - mean[i]) / std[i]
        return im

    @property
    def output_shape(self):
        return (self.input_shape[0], self.model.num_features)


class DINOv2Encoder(EncoderModule):
    # Per-channel mean and standard deviation (in RGB order)
    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def __init__(self, input_shape: Tuple[int, int, int, int]):
        super().__init__(input_shape)
        self.input_shape = input_shape
        assert input_shape[1] == 3, "DINOv2 only supports channel of size 3"
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14_lc"
        ).backbone
        self.model.eval()

    def forward(self, x: torch.Tensor):
        assert (
            self.input_shape == x.shape[1:]
        ), f"expected input shape {self.input_shape} but got {x.shape[1:]}"
        B, V = x.shape[:2]
        with torch.no_grad():
            outs = []
            for _x in x.unbind(1):
                _x = F.interpolate(
                    _x, size=(224, 224), mode="bilinear", align_corners=False
                )
                _x = self._color_norm(_x / 255.0)
                _x = self.model(_x)
                outs.append(_x.view(B, -1))
            fused = torch.stack(outs, 1)
        return fused

    def _color_norm(self, im):
        """Performs per-channel normalization."""
        mean = DINOv2Encoder._MEAN
        std = DINOv2Encoder._STD
        for i in range(3):
            im[..., i, :, :] = (im[..., i, :, :] - mean[i]) / std[i]
        return im

    @property
    def output_shape(self):
        return (self.input_shape[0], self.model.embed_dim)


class R3MEncoder(EncoderModule):
    # Per-channel mean and standard deviation (in RGB order)
    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        model: str,
    ):
        super().__init__(input_shape)
        assert input_shape[1] == 3, "R3M only supports channel of size 3"

        try:
            import ssl

            ssl._create_default_https_context = ssl._create_unverified_context
            from r3m import load_r3m

        except ImportError:
            raise ImportError(
                "Please run: pip install git+https://github.com/ir413/mvp"
            )

        if model == "r3m_resnet18":
            model = load_r3m("resnet18")
        elif model == "r3m_resnet34":
            model = load_r3m("resnet34")
        elif model == "r3m_resnet50":
            model = load_r3m("resnet50")
        else:
            raise ValueError(model)
        self.num_features = model.module.outdim
        self.model = model.module.convnet
        self.model.eval()

    def forward(self, x: torch.Tensor):
        assert (
            self.input_shape == x.shape[1:]
        ), f"expected input shape {self.input_shape} but got {x.shape[1:]}"
        B, V = x.shape[:2]
        with torch.no_grad():
            outs = []
            for _x in x.unbind(1):
                _x = F.interpolate(
                    _x, size=(224, 224), mode="bilinear", align_corners=False
                )
                _x = self._color_norm(_x / 255.0)
                _x = self.model(_x)
                outs.append(_x.view(B, -1))
            fused = torch.stack(outs, 1)
        return fused

    def _color_norm(self, im):
        """Performs per-channel normalization."""
        mean = ResNetEncoder._MEAN
        std = ResNetEncoder._STD
        for i in range(3):
            im[..., i, :, :] = (im[..., i, :, :] - mean[i]) / std[i]
        return im

    @property
    def output_shape(self):
        return (self.input_shape[0], self.num_features)


class AttentionPooling(nn.Module):
    """
    Attention pooling layer that learns to weight different patches
    More expressive than simple average or CLS token pooling
    """
    def __init__(self, embed_dim: int, num_heads: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Query vector for attention pooling
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Multi-head attention for pooling
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: [B, N, D] where N is sequence length, D is embedding dimension
        Returns:
            pooled: [B, D] pooled representation
        """
        B, N, D = x.shape
        
        # Expand query to batch size
        query = self.query.expand(B, -1, -1)  # [B, 1, D]
        
        # Apply attention pooling
        pooled, attention_weights = self.attention(
            query=query,    # [B, 1, D]
            key=x,         # [B, N, D] 
            value=x        # [B, N, D]
        )
        
        # Apply layer norm and squeeze
        pooled = self.norm(pooled.squeeze(1))  # [B, D]
        
        return pooled


class EncoderMultiViewVisionTransformer(EncoderModule):
    # Per-channel mean and standard deviation (in RGB order)
    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        input_shape: Tuple[int, int, int, int] = None,
        shape_meta: dict = None,
        patch_size: int=14,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 3,
        decoder_num_heads: int = 4,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        conv_embed: bool = False,
        reward_pred: bool = True,
        # MultiImageObsEncoder compatibility parameters
        resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
        crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
        random_crop: bool=True,
        use_agent_pos: bool=True,
        # New parameter for latent vector output
        latent_dim: int = 128,
        # Pooling method parameter
        pooling_method: str = "attention",  # "attention", "cls", "average"
        attention_pool_heads: int = 1,  # Number of heads for attention pooling
        # Latent projection control
        use_latent_projection: bool = False,  # Whether to use latent projection layer
    ):
        """
        This class is an implementation of Encoder class in
        Multi-View Masked Autoencoder (MV-MAE; https://arxiv.org/abs/2302.02408)
        
        Now supports both tensor input format and dictionary input format (MultiImageObsEncoder style)

        Args:
            input_shape (Tuple[int, int, int, int], optional): V,C,H,W where V is the number of
                viewpoints and C is the channel of viewpoint. Use this for tensor input.
                If shape_meta is provided, this parameter will be ignored and auto-constructed.
            shape_meta (dict, optional): Dictionary containing observation shape metadata.
                Use this for dictionary input format. If provided, input_shape will be auto-constructed.
            patch_size (int): Patch size. This should be the power of 2.
            embed_dim (int, optional): Embedding dimension of the encoder.
            depth (int, optional): Depth of the encoder.
            num_heads (int, optional): Number of heads in the encoder.
            decoder_embed_dim (int, optional): Embedding dimension of the decoder.
            decoder_depth (int, optional): Depth of the decoder.
            decoder_num_heads (int, optional): Number of heads in the decoder.
            mlp_ratio (float, optional): MLP ratio for linear layers in Transformer.
            norm_layer (nn.Module, optional): Type of layernorm in Transformer.
            conv_embed (bool, optional): Use convolutional feature masking
            reward_pred (bool, optional): Use reward prediction as auxiliary objective
            resize_shape: Target resize shape for images
            crop_shape: Target crop shape for images
            random_crop: Whether to use random cropping
            use_agent_pos: Whether to include agent position in output
            latent_dim: Dimension for latent vector output (only used when use_latent_projection=True)
            pooling_method: Pooling method ("attention", "cls", "average")
            attention_pool_heads: Number of heads for attention pooling
            use_latent_projection: Whether to use latent projection layer for fixed-dimension output
        """
        # Determine if using dictionary input format
        self.use_dict_input = shape_meta is not None
        self.use_agent_pos = use_agent_pos
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.latent_dim = latent_dim
        self.pooling_method = pooling_method.lower()
        self.attention_pool_heads = attention_pool_heads
        self.use_latent_projection = use_latent_projection
        
        # Validate pooling method
        valid_pooling_methods = ["attention", "cls", "average"]
        if self.pooling_method not in valid_pooling_methods:
            raise ValueError(f"pooling_method must be one of {valid_pooling_methods}, got {pooling_method}")
        
        if self.use_dict_input:
            # Process shape_meta to automatically construct input_shape
            rgb_keys = []
            low_dim_keys = []
            key_shape_map = {}
            key_transform_map = {}  # Initialize as regular dict first
            
            obs_shape_meta = shape_meta['obs']
            for key, attr in obs_shape_meta.items():
                shape = tuple(attr['shape'])
                type = attr.get('type', 'low_dim')
                key_shape_map[key] = shape
                
                if type == 'rgb':
                    rgb_keys.append(key)
                    
                    # Configure transforms for this key
                    input_shape_key = shape
                    this_resizer = nn.Identity()
                    if resize_shape is not None:
                        if isinstance(resize_shape, dict):
                            h, w = resize_shape[key]
                        else:
                            h, w = resize_shape
                        this_resizer = torchvision.transforms.Resize(size=(h,w))
                        input_shape_key = (shape[0], h, w)
                    
                    # Configure randomizer
                    this_randomizer = nn.Identity()
                    if crop_shape is not None:
                        if isinstance(crop_shape, dict):
                            h, w = crop_shape[key]
                        else:
                            h, w = crop_shape
                        if random_crop:
                            # Import CropRandomizer if needed
                            try:
                                from rl_100.model.vision.crop_randomizer import CropRandomizer
                                this_randomizer = CropRandomizer(
                                    input_shape=input_shape_key,
                                    crop_height=h,
                                    crop_width=w,
                                    num_crops=1,
                                    pos_enc=False
                                )
                            except ImportError:
                                this_randomizer = torchvision.transforms.CenterCrop(size=(h,w))
                        else:
                            this_randomizer = torchvision.transforms.CenterCrop(size=(h,w))
                    
                    this_transform = nn.Sequential(this_resizer, this_randomizer)
                    key_transform_map[key] = this_transform
                    
                elif type == 'low_dim':
                    low_dim_keys.append(key)
            
            rgb_keys = sorted(rgb_keys)
            low_dim_keys = sorted(low_dim_keys)
            
            # Store for later use (don't assign ModuleDict yet)
            self.rgb_keys = rgb_keys
            self.low_dim_keys = low_dim_keys
            self.key_shape_map = key_shape_map
            self._key_transform_map_dict = key_transform_map  # Store temporarily
            self.shape_meta = shape_meta
            
            # Automatically construct input_shape from shape_meta
            if len(rgb_keys) > 0:
                first_key = rgb_keys[0]
                test_shape = key_shape_map[first_key]
                
                # Apply resize
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[first_key]
                    else:
                        h, w = resize_shape
                    test_shape = (test_shape[0], h, w)
                
                # Apply crop
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[first_key]
                    else:
                        h, w = crop_shape
                    test_shape = (test_shape[0], h, w)
                
                # Auto-construct input_shape for ViT: (V, C, H, W)
                input_shape = (len(rgb_keys), test_shape[0], test_shape[1], test_shape[2])
            else:
                raise ValueError("No RGB observations found in shape_meta")
        else:
            # Traditional tensor input format - input_shape is required
            if input_shape is None:
                raise ValueError("input_shape must be provided when shape_meta is not specified")
        
        # Initialize parent class
        super().__init__(input_shape)
        
        # Now we can safely assign ModuleDict after super().__init__()
        if self.use_dict_input:
            self.key_transform_map = nn.ModuleDict(self._key_transform_map_dict)
            del self._key_transform_map_dict  # Clean up temporary storage
        num_views = input_shape[0]
        in_chans = input_shape[1]
        img_size = input_shape[2:4]

        # --- Normalization Override ---
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # --- Encoder specifics ---
        embed_cls = MultiViewConvEmbed if conv_embed else MultiViewPatchEmbed
        self.patch_embed = embed_cls(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Output shape
        self._output_shape = (num_views, num_patches * embed_dim)
        self._num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        self.view_embed = nn.Parameter(torch.zeros(1, num_views, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self._norm = norm_layer(embed_dim)

        # --- Pooling layer ---
        if self.pooling_method == "attention":
            self.pooling_layer = AttentionPooling(embed_dim, self.attention_pool_heads)
            pooled_feature_dim = embed_dim
        elif self.pooling_method == "cls":
            # CLS token pooling - no additional layer needed
            self.pooling_layer = None
            pooled_feature_dim = embed_dim
        elif self.pooling_method == "average":
            # Average pooling - no additional layer needed
            self.pooling_layer = None
            pooled_feature_dim = embed_dim

        # --- Latent projection layer for fixed-dimension output ---
        # Only create latent projection if use_latent_projection is True
        if self.use_latent_projection:
            # Calculate the total feature dimension before projection
            vit_feature_dim = num_views * pooled_feature_dim  # Now using pooled features
            
            # Add low-dim features if using dict input
            if self.use_agent_pos and len(self.low_dim_keys) > 0:
                # We'll calculate this lazily when first needed
                self._low_dim_size = None
                self.latent_projection = None
            else:
                # Pure ViT output
                self._low_dim_size = 0
                self.latent_projection = nn.Sequential(
                    nn.Linear(vit_feature_dim, latent_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(latent_dim * 2, latent_dim)
                )
        else:
            # No latent projection
            self.latent_projection = None
            self._low_dim_size = 0

        # --- Decoder specifics ---
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        self.decoder_view_embed = nn.Parameter(
            torch.zeros(1, num_views, 1, decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch

        # --- Reward specifics ---
        self.reward_pred = False
        if reward_pred:
            self.reward_pred = True
            self.reward_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_reward_pred = nn.Linear(decoder_embed_dim, 1, bias=True)

        # Initialize all the weights
        self.initialize_weights()
        
        # Calculate output shape for dictionary input - do this after initialization
        if self.use_dict_input:
            # We'll calculate this lazily when first needed
            self._dict_output_dim = None

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.patch_embed.grid_size,
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            self.patch_embed.grid_size,
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02)
        # as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        torch.nn.init.normal_(self.view_embed, std=0.02)
        torch.nn.init.normal_(self.decoder_view_embed, std=0.02)
        if self.reward_pred:
            torch.nn.init.normal_(self.reward_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _preprocess_input(self, input_data):
        """
        统一的输入预处理模块
        检测输入类型并进行相应的预处理
        
        Args:
            input_data: 可以是张量 (B, V, C, H, W) 或字典格式
            
        Returns:
            processed_imgs: 预处理后的图像张量 (B, V, C, H, W)
            additional_features: 额外的低维特征列表
            batch_size: 批次大小
        """
        if isinstance(input_data, dict):
            # 字典输入格式
            return self._preprocess_dict_input(input_data)
        else:
            # 张量输入格式
            return self._preprocess_tensor_input(input_data)
    
    def _preprocess_dict_input(self, obs_dict):
        """处理字典格式输入"""
        batch_size = None
        additional_features = []
        
        # 处理RGB图像
        if len(self.rgb_keys) > 0:
            imgs = []
            for key in self.rgb_keys:
                img = obs_dict[key]
                if img.shape[1] != 3:
                    img = einops.rearrange(img, "b h w c -> b c h w")
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            
            # 堆叠成多视角格式: (B, V, C, H, W)
            processed_imgs = torch.stack(imgs, dim=1)
        else:
            raise ValueError("No RGB observations found")
        
        # 处理低维数据
        if self.use_agent_pos:
            for key in self.low_dim_keys:
                data = obs_dict[key]
                if batch_size is None:
                    batch_size = data.shape[0]
                else:
                    assert batch_size == data.shape[0]
                assert data.shape[1:] == self.key_shape_map[key]
                additional_features.append(data)
        
        return processed_imgs, additional_features, batch_size
    
    def _preprocess_tensor_input(self, x):
        """处理张量格式输入"""
        batch_size = x.shape[0]
        additional_features = []
        return x, additional_features, batch_size

    def forward(self, x):
        if self.use_dict_input:
            return self.forward_dict(x)
        else:
            return self.forward_tensor(x)
    
    def _initialize_latent_projection(self):
        """Lazily initialize the latent projection layer for dictionary input"""
        if self.use_dict_input and self.use_latent_projection and self.latent_projection is None:
            # Calculate total feature dimension using pooled features
            pooled_feature_dim = self.blocks[0].norm1.normalized_shape[0]  # embed_dim
            vit_feature_dim = self.input_shape[0] * pooled_feature_dim
            
            # Calculate low-dim feature size
            if self.use_agent_pos and len(self.low_dim_keys) > 0:
                low_dim_size = sum(np.prod(self.key_shape_map[key]) for key in self.low_dim_keys)
            else:
                low_dim_size = 0
            
            total_feature_dim = vit_feature_dim + low_dim_size
            self._low_dim_size = low_dim_size
            
            # Create projection layer
            self.latent_projection = nn.Sequential(
                nn.Linear(total_feature_dim, self.latent_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.latent_dim * 2, self.latent_dim)
            ).to(next(self.parameters()).device)
    
    def _apply_pooling(self, encoded_features):
        """
        Apply pooling to encoded features from ViT
        Args:
            encoded_features: [B, V*L+1, D] (includes CLS token)
        Returns:
            pooled_features: [B, V, D] pooled features per view
        """
        B = encoded_features.shape[0]
        V = self.input_shape[0]  # number of views
        L = self._num_patches  # patches per view
        D = encoded_features.shape[-1]  # embed_dim
        
        if self.pooling_method == "cls":
            # Use CLS token (first token)
            cls_feature = encoded_features[:, 0, :]  # [B, D]
            # Replicate for each view (this is a simplification, in practice you might want separate CLS tokens per view)
            pooled_features = cls_feature.unsqueeze(1).expand(B, V, D)  # [B, V, D]
            
        elif self.pooling_method == "average":
            # Remove CLS token and reshape to separate views
            patch_features = encoded_features[:, 1:, :]  # [B, V*L, D]
            patch_features = patch_features.reshape(B, V, L, D)  # [B, V, L, D]
            # Average pool over patches for each view
            pooled_features = patch_features.mean(dim=2)  # [B, V, D]
            
        elif self.pooling_method == "attention":
            # Apply attention pooling to each view separately
            patch_features = encoded_features[:, 1:, :]  # [B, V*L, D]
            patch_features = patch_features.reshape(B, V, L, D)  # [B, V, L, D]
            
            pooled_per_view = []
            for v in range(V):
                view_features = patch_features[:, v, :, :]  # [B, L, D]
                pooled_view = self.pooling_layer(view_features)  # [B, D]
                pooled_per_view.append(pooled_view)
            
            pooled_features = torch.stack(pooled_per_view, dim=1)  # [B, V, D]
        
        return pooled_features
    
    def forward_tensor(self, x):
        """Forward method for tensor input - outputs latent vector or pooled features"""
        processed_imgs, additional_features, batch_size = self._preprocess_input(x)
        
        # ViT处理
        B, V = processed_imgs.shape[:2]
        processed_imgs = self._color_norm(processed_imgs / 255.0)
        encoded, _, _ = self.forward_encoder(processed_imgs, mask_ratio=0.0, num_mask_views=0)
        
        # Apply pooling to get per-view features
        pooled_features = self._apply_pooling(encoded)  # [B, V, D]
        
        if self.use_latent_projection:
            # Use latent projection to get fixed-dimension output
            # Flatten to [B, V*D]
            vit_features = pooled_features.reshape(B, -1)
            
            # 合并所有特征
            all_features = [vit_features] + additional_features
            if len(all_features) > 0:
                combined_features = torch.cat(all_features, dim=-1)
            else:
                combined_features = vit_features
            
            # 投影到latent空间
            latent_vector = self.latent_projection(combined_features)
            return latent_vector
        else:
            # Return pooled features directly
            return pooled_features
    
    def forward_dict(self, obs_dict):
        """Forward method for dictionary input - outputs latent vector or pooled features"""
        # Initialize projection layer if needed
        if self.use_latent_projection:
            self._initialize_latent_projection()
        
        processed_imgs, additional_features, batch_size = self._preprocess_input(obs_dict)
        
        # ViT处理
        B, V = processed_imgs.shape[:2]
        processed_imgs = self._color_norm(processed_imgs / 255.0)
        encoded, _, _ = self.forward_encoder(processed_imgs, mask_ratio=0.0, num_mask_views=0)
        
        # Apply pooling to get per-view features
        pooled_features = self._apply_pooling(encoded)  # [B, V, D]
        
        if self.use_latent_projection:
            # Use latent projection to get fixed-dimension output
            # Flatten to [B, V*D]
            vit_features = pooled_features.reshape(B, -1)
            
            # 合并所有特征
            all_features = [vit_features] + additional_features
            if len(all_features) > 0:
                combined_features = torch.cat(all_features, dim=-1)
            else:
                combined_features = vit_features
            
            # 投影到latent空间
            latent_vector = self.latent_projection(combined_features)
            return latent_vector
        else:
            # Return pooled features directly
            # If there are additional features, concatenate them
            if additional_features:
                # Flatten pooled features and concatenate with additional features
                vit_features = pooled_features.reshape(B, -1)
                all_features = [vit_features] + additional_features
                combined_features = torch.cat(all_features, dim=-1)
                return combined_features
            else:
                return pooled_features

    def forward_encoder(self, x, mask_ratio, num_mask_views):
        # x: [B, V, C, H, W]
        x = self.patch_embed(x)  # Embed to [B, V, L, C]

        # Add pos embed w/o cls token: [1, 1, L, C]
        x = x + self.pos_embed[:, 1:, :].unsqueeze(1)

        # Add view embed: [1, V, 1, C]
        x = x + self.view_embed

        # masking: length -> length * mask_ratio
        if mask_ratio != 0.0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio, num_mask_views)
        else:
            # reshape to [B, V*L, D]
            x = x.reshape([x.shape[0], -1, x.shape[-1]])
            mask, ids_restore = None, None

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self._norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        if ids_restore is not None:
            # Append mask tokens to sequence -> [N, V * L, D]
            mask_tokens = self.mask_token.repeat(
                x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
            )
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(
                x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
            )  # unshuffle
        else:
            x_ = x[:, 1:]

        # [N, V*L, D] -> [N, V, L, D]
        L = self.decoder_pos_embed.shape[1] - 1
        x_ = torch.stack(torch.split(x_, L, 1), 1)

        # Add pos embed w/o cls token: [1, 1, L, C]
        x_ = x_ + self.decoder_pos_embed[:, 1:, :].unsqueeze(1)

        # Add view embed: [1, V, 1, C]
        x_ = x_ + self.decoder_view_embed

        # [N, V, L, D] -> [N, V*L, D]
        x_ = torch.reshape(x_, [x_.shape[0], -1, x_.shape[-1]])

        # Append cls token
        cls_token = x[:, :1, :] + self.decoder_pos_embed[:, :1, :]
        x = torch.cat([cls_token, x_], dim=1)

        # Append reward token, if required
        if self.reward_pred:
            reward_token = self.reward_token.repeat(x.shape[0], 1, 1)
            x = torch.cat([x, reward_token], dim=1)

        # Apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        output = {}
        # Reward pred
        if self.reward_pred:
            reward_x = x[:, -1, :]
            reward_x = self.decoder_reward_pred(reward_x)  # [N, 1]
            output["reward"] = reward_x
            x = x[:, :-1, :]

        # Predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        # [N, V*L, D] -> [N, V, L, D]
        x = torch.stack(torch.split(x, L, 1), 1)
        output["image"] = x
        return output

    def forward_loss(self, imgs, preds):
        """
        imgs: [N, V, 3, H, W]
        preds: [N, V, L, p*p*3]
        """

        V = imgs.shape[1]
        loss = 0.0
        for v in range(V):
            target = self.patchify(imgs[:, v])
            pred = preds[:, v]
            loss_v = (pred - target) ** 2
            loss = loss + loss_v.mean()
        loss /= float(V)
        return loss

    def forward_reward_loss(self, reward, reward_preds):
        """
        reward: [N, 1]
        reward_preds: [N, 1]
        """
        x = reward_preds
        loss = ((reward - x) ** 2).mean()
        return loss

    def calculate_loss(
        self, imgs_or_obs_dict, reward=None, mask_ratio=0.9, num_mask_views=1, return_mets=False
    ):
        """
        计算损失，支持张量和字典输入格式
        
        Args:
            imgs_or_obs_dict: 可以是图像张量或观测字典
            reward: 奖励信号（可选）
            mask_ratio: 遮罩比例
            num_mask_views: 遮罩视角数量
            return_mets: 是否返回指标
        """
        mets = {}
        
        # 预处理输入
        processed_imgs, additional_features, batch_size = self._preprocess_input(imgs_or_obs_dict)
        
        # 标准化图像
        processed_imgs = self._color_norm(processed_imgs / 255.0)
        
        # 前向编码
        latent, _, ids_restore = self.forward_encoder(processed_imgs, mask_ratio, num_mask_views)
        output = self.forward_decoder(latent, ids_restore)
        
        # 标准MAE损失
        preds = output["image"]
        loss = self.forward_loss(processed_imgs, preds)
        mets["mae_image_loss"] = loss.item()
        
        # 奖励损失（如果需要）
        if self.reward_pred and reward is not None:
            reward_preds = output["reward"]
            reward_loss = self.forward_reward_loss(reward, reward_preds)
            mets["mae_reward_loss"] = reward_loss.item()
            loss = loss + reward_loss
            
        if return_mets:
            out = (loss, mets)
        else:
            out = loss
        return out

    def report(self, imgs_or_obs_dict, mask_ratio, num_mask_views, pre_shape):
        """
        生成可视化报告，支持张量和字典输入格式
        
        Args:
            imgs_or_obs_dict: 可以是图像张量或观测字典
            mask_ratio: 遮罩比例
            num_mask_views: 遮罩视角数量
            pre_shape: 预处理形状
        """
        # 预处理输入
        processed_imgs, additional_features, batch_size = self._preprocess_input(imgs_or_obs_dict)
        
        # 标准化图像
        processed_imgs = self._color_norm(processed_imgs / 255.0)
        
        # 前向传播
        latent, _, ids_restore = self.forward_encoder(processed_imgs, mask_ratio, num_mask_views)
        preds = self.forward_decoder(latent, ids_restore)["image"]

        with torch.no_grad():
            V = processed_imgs.shape[1]

            # 构建可视化图像
            out = []
            for v in range(V):
                img = processed_imgs[:, v]
                pred = self.unpatchify(preds[:, v])
                out.append(torch.cat([img, pred], -2))  # height-wise concat
            out = torch.cat(out, -1)  # width-wise concat for view aggregation

            # [B*T, C, H*2, W*V]
            # pre_shape: [B, T]
            out = out.reshape(pre_shape + out.shape[1:])
            out = torch.cat(
                torch.unbind(out, 0), -1
            )  # width-wise batch aggregation, #[T, C, H*2, W*V*B]
            out = self._color_denorm(out)
            out = torch.clip(out, 0.0, 1.0)
            out = (out * 255.0).byte()
            out = out.permute(0, 2, 3, 1).cpu().numpy()  # B H W C
        return {"mae_visualization": {"video": out, "fps": 4}}
    
    def _calculate_dict_output_shape(self):
        """Calculate output shape for dictionary input format"""
        if not self.use_dict_input or self._dict_output_dim is not None:
            return
            
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        
        # Get device and dtype from model parameters
        try:
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
        except StopIteration:
            device = torch.device('cpu')
            dtype = torch.float32
        
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=dtype,
                device=device)
            example_obs_dict[key] = this_obs
        with torch.no_grad():
            example_output = self.forward_dict(example_obs_dict)
            self._dict_output_dim = example_output.shape[1:]
    
    @property
    def dict_output_shape(self):
        """Get output shape for dictionary input"""
        if self.use_dict_input:
            if self._dict_output_dim is None:
                self._calculate_dict_output_shape()
            return self._dict_output_dim
        else:
            return None

    def random_masking(self, x, mask_ratio, num_mask_views):
        """
        Perform per-sample random view masking and per-sample random masking
        For view masking, we randomly sample viewpoints and masking whole viewpoints.
        Per-sample random masking is done for remaining viewpoints after view masking.
        Per-sampling shuffling is done by argsort random noise

        Inputs:
        - x: [N, V, L, D]

        Outputs:
        - [N, LEN_KEEP, D]
        """
        if num_mask_views != 0:
            x_masked, mask, ids_restore = self._random_view_masking(
                x, mask_ratio, num_mask_views
            )
        else:
            x_masked, mask, ids_restore = self._random_uniform_masking(x, mask_ratio)

        return x_masked, mask, ids_restore

    def _random_view_masking(self, x, mask_ratio, num_mask_views):
        N, V, L, D = x.shape
        assert num_mask_views >= 1 and V >= 1

        # Construct noises for view masking
        mask_view_list, noises = [], []
        for v in range(V):
            # This decides whether to mask this view or not per-sample in minibatch
            view_noise = torch.rand(N, 1, device=x.device)

            if v == 0:
                mask_view = view_noise > 0.5
                no_mask_view = ~mask_view
            else:
                curr_mask_views = torch.sum(
                    torch.cat(mask_view_list, 1), 1, keepdim=True
                )

                # M = num_mask_views
                # Find samples whose M views are already masked
                done_mask_views = curr_mask_views == num_mask_views

                # Find samples whose views should be masked to meet the number M.
                # e.g., if 1 of 3 is not masked, 2 remaining ones should be masked.
                num_should_mask_views = num_mask_views - curr_mask_views
                num_remaining_views = V - v
                must_mask_views = num_should_mask_views == num_remaining_views

                # Find samples that (i) should be masked or (ii) randomly view-masked
                mask_view = must_mask_views | (~must_mask_views & (view_noise > 0.5))

                # Filter out samples that are alredy masked
                mask_view = ~done_mask_views & mask_view
                no_mask_view = ~mask_view

            # Noises that will be used for samples whose v-viewpoint is not masked
            uniform_noise = torch.rand(N, L, device=x.device)
            # Noises that will be used for samples whose v-viewpoint is masked
            # This is set as 1 because noises with high values will be masked
            view_masked_noise = torch.ones(N, L, device=x.device) + 1e-2

            # Construct per-sample noises
            dtype = uniform_noise.dtype
            noise = (
                mask_view.to(dtype) * view_masked_noise
                + no_mask_view.to(dtype) * uniform_noise
            )

            mask_view_list.append(mask_view)
            noises.append(noise)
        noise = torch.cat(noises, 1)

        # We should adjust the effective masking ratio with view-masking
        # For instance, when we use view-masking of 1 with 2 viewpoints
        # 50% will be already masked with view-masking,
        # 40% remaining masks will be sampled from remaining 1 viewpoint,
        # so 90% masking ratio becomes effectively 80% in terms of a single viewpoint
        # so we have to adjust the effective masking ratio. This ensures that
        # uniform masking with mask_ratio is applied to unmasked viewpoint
        mask_ratio = mask_ratio + (1.0 - mask_ratio) * (num_mask_views / V)
        len_keep = int(V * L * (1 - mask_ratio))

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # make input be [N, V * L, D]
        x = x.reshape([N, V * L, D])
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, V * L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # make mask be [N, V, L]
        mask = mask.reshape([N, V, L])

        return x_masked, mask, ids_restore

    def _random_uniform_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, V, L, D]
        """
        # Convert to [N, V * L, D]
        N, V, L, D = x.shape
        x = torch.reshape(x, [N, V * L, D])

        VL = V * L
        len_keep = int(VL * (1 - mask_ratio))

        noise = torch.rand(N, VL, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, VL], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask = mask.reshape([N, V, L])

        return x_masked, mask, ids_restore

    def _color_norm(self, im):
        """Performs per-channel normalization."""
        mean = EncoderMultiViewVisionTransformer._MEAN
        std = EncoderMultiViewVisionTransformer._STD
        for i in range(3):
            im[..., i, :, :] = (im[..., i, :, :] - mean[i]) / std[i]
        return im

    def _color_denorm(self, im):
        """Performs per-channel denormalization."""
        mean = EncoderMultiViewVisionTransformer._MEAN
        std = EncoderMultiViewVisionTransformer._STD
        for i in range(3):
            im[..., i, :, :] = im[..., i, :, :] * std[i] + mean[i]
        return im


    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        c = imgs.shape[1]
        h, w = self.patch_embed.grid_size
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h, w = self.patch_embed.grid_size
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    @torch.no_grad()
    def output_shape(self):
        """Return output shape as scalar dimension for compatibility with other encoders"""
        if self.use_latent_projection:
            # Return latent vector output dimension
            return self.latent_dim
        else:
            # Calculate output features dynamically
            if self.use_dict_input:
                # For dictionary input format
                if self._dict_output_dim is None:
                    self._calculate_dict_output_shape()
                # Return total feature dimension
                if len(self._dict_output_dim) == 1:
                    return self._dict_output_dim[0]
                else:
                    # Flatten the output shape to get total dimension
                    import numpy as np
                    return int(np.prod(self._dict_output_dim))
            else:
                # For tensor input format
                pooled_feature_dim = self.blocks[0].norm1.normalized_shape[0]  # embed_dim
                return self.input_shape[0] * pooled_feature_dim  # V * D

    @property
    def num_patches(self):
        # number of patches per each view
        return self._num_patches

    @property 
    def mae_output_shape(self):
        """Return original MAE output shape for training purposes"""
        return self._output_shape

    @torch.no_grad()
    def calculate_output_shape(self):
        """
        Calculate output shape by running forward pass with example input.
        Similar to MultiImageObsEncoder.output_shape() functionality.
        """
        if self.use_dict_input:
            # For dictionary input format
            example_obs_dict = dict()
            obs_shape_meta = self.shape_meta['obs']
            batch_size = 1
            
            # Get device and dtype from model parameters
            try:
                device = next(self.parameters()).device
                dtype = next(self.parameters()).dtype
            except StopIteration:
                device = torch.device('cpu')
                dtype = torch.float32
            
            for key, attr in obs_shape_meta.items():
                shape = tuple(attr['shape'])
                this_obs = torch.zeros(
                    (batch_size,) + shape, 
                    dtype=dtype,
                    device=device)
                example_obs_dict[key] = this_obs
            
            example_output = self.forward(example_obs_dict)
            output_shape = example_output.shape[1:]
            return output_shape
        else:
            # For tensor input format
            batch_size = 1
            
            # Get device and dtype from model parameters
            try:
                device = next(self.parameters()).device
                dtype = next(self.parameters()).dtype
            except StopIteration:
                device = torch.device('cpu')
                dtype = torch.float32
            
            # Create example tensor input
            example_input = torch.zeros(
                (batch_size,) + self.input_shape,
                dtype=dtype,
                device=device
            )
            
            example_output = self.forward(example_input)
            output_shape = example_output.shape[1:]
            return output_shape