import os
from typing import Optional, Union

import torch
from huggingface_hub import hf_hub_download

from .sr_pipeline import KandiSuperResPipeline
from KandiSuperRes.model.unet import UNet


def get_sr_model(
    device: Union[str, torch.device],
    weights_path: Optional[str] = None,
    dtype: Union[str, torch.dtype] = torch.float16
) -> (UNet, Optional[dict], Optional[torch.Tensor]):
    unet = UNet(
        init_channels=128,
        model_channels=128,
        num_channels=3,
        time_embed_dim=512,
        groups=32,
        dim_mult=(1, 2, 4, 8),
        num_resnet_blocks=(2,4,8,8),
        add_cross_attention=(False, False, False, False),
        add_self_attention=(False, False, False, False),
        feature_pooling_type='attention',
        lowres_cond =True,
        efficient=False
    )

    if weights_path:
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        unet_state_dict = {
            key.replace('unet.', ''): value
            for key, value in state_dict['callbacks']['EMA']['ema_state_dict'].items()  if 'unet' in key
        }

        unet.load_state_dict(unet_state_dict)

    unet.to(device=device, dtype=dtype).eval()
    return unet


def get_SR_pipeline(
    device: Union[str, torch.device],
    fp16: bool = True,
    model_path: str = None,
) -> KandiSuperResPipeline:
    
    dtype = torch.float16 if fp16 else torch.float32
    sr_model = get_sr_model(device, model_path, dtype=dtype)
    return KandiSuperResPipeline(device, dtype, sr_model)