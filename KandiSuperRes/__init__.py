import torch
from typing import Optional, Union
from huggingface_hub import hf_hub_download

from .sr_pipeline import KandiSuperResPipeline
from KandiSuperRes.model.unet import UNet
from KandiSuperRes.model.unet_sr import UNet as UNet_sr
from KandiSuperRes.movq import MoVQ


def get_sr_model(
    device: Union[str, torch.device],
    weights_path: Optional[str] = None,
    dtype: Union[str, torch.dtype] = torch.float16
) -> (UNet_sr, Optional[dict], Optional[torch.Tensor]):
    unet = UNet_sr(
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
        lowres_cond =True
    )

    if weights_path:
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        try:
            unet.load_state_dict(state_dict['unet'])
        except:
            unet.load_state_dict(state_dict)
    unet.to(device=device, dtype=dtype).eval()
    return unet
    

def get_T2I_unet(
    device: Union[str, torch.device],
    weights_path: Optional[str] = None, 
    dtype: Union[str, torch.dtype] = torch.float32,
) -> (UNet, Optional[torch.Tensor], Optional[dict]):
    unet = UNet(
        model_channels=384,
        num_channels=4,
        init_channels=192,
        time_embed_dim=1536,
        context_dim=4096,
        groups=32,
        head_dim=64,
        expansion_ratio=4,
        compression_ratio=2,
        dim_mult=(1, 2, 4, 8),
        num_blocks=(3, 3, 3, 3),
        add_cross_attention=(False, True, True, True),
        add_self_attention=(False, True, True, True),
    )

    null_embedding = None
    if weights_path:
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        null_embedding = state_dict['null_embedding']
        unet.load_state_dict(state_dict['unet'])
    
    unet.to(device=device, dtype=dtype).eval()
    return unet, null_embedding


def get_movq(
    device: Union[str, torch.device],
    weights_path: Optional[str] = None,
    dtype: Union[str, torch.dtype] = torch.float32,
) -> MoVQ:
    generator_config = {
        'double_z': False,
        'z_channels': 4,
        'resolution': 256,
        'in_channels': 3,
        'out_ch': 3,
        'ch': 256,
        'ch_mult': [1, 2, 2, 4],
        'num_res_blocks': 2,
        'attn_resolutions': [32],
        'dropout': 0.0,
        'tile_sample_min_size': 1024,
        'tile_overlap_factor_enc': 0.0,
        'tile_overlap_factor_dec': 0.25,
        'use_tiling': True
    }
    movq = MoVQ(generator_config)

    if weights_path:
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        movq.load_state_dict(state_dict)

    movq.to(device=device, dtype=dtype).eval()
    return movq


def get_SR_pipeline(
    device: Union[str, torch.device],
    fp16: bool = True,
    flash: bool = True,
    scale: int = 2,
    cache_dir: str = '/tmp/KandiSuperRes/',
    movq_path: str = None,
    refiner_path: str = None,
    unet_sr_path: str = None,
) -> KandiSuperResPipeline:
    
    if flash:
        if scale == 2:
            device_map = {
            'movq': device, 'refiner': device, 'sr_model': device
            } 
            dtype = torch.float16 if fp16 else torch.float32 
            dtype_map = {
                'movq': torch.float32, 'refiner': dtype, 'sr_model': dtype
            }
            if movq_path is None:
                print('Download movq weights')
                movq_path = hf_hub_download(
                    repo_id="ai-forever/Kandinsky3.1", filename='weights/movq.pt', cache_dir=cache_dir
                )
            if refiner_path is None:
                print('Download refiner weights')
                refiner_path = hf_hub_download(
                    repo_id="ai-forever/Kandinsky3.1", filename='weights/kandinsky3_flash.pt', cache_dir=cache_dir
                )
            if unet_sr_path is None:
                print('Download KandiSuperRes Flash weights')
                unet_sr_path = hf_hub_download(
                    repo_id="ai-forever/KandiSuperRes", filename='KandiSuperRes_flash_x2.pt', cache_dir=cache_dir
                )
            sr_model = get_sr_model(device_map['sr_model'], unet_sr_path, dtype=dtype_map['sr_model'])
            movq = get_movq(device_map['movq'], movq_path, dtype=dtype_map['movq'])
            refiner, _ = get_T2I_unet(device_map['refiner'], refiner_path, dtype=dtype_map['refiner'])
            return KandiSuperResPipeline(
                scale, device_map, dtype_map, flash, sr_model, movq, refiner
            )        
        else:
            print('Flash model for x4 scale is not implemented.')
    else:
        if unet_sr_path is None:
            if scale == 4:
                unet_sr_path = hf_hub_download(
                    repo_id="ai-forever/KandiSuperRes", filename='KandiSuperRes.ckpt', cache_dir=cache_dir
                )
            elif scale == 2:
                unet_sr_path = hf_hub_download(
                    repo_id="ai-forever/KandiSuperRes", filename='KandiSuperRes_x2.ckpt', cache_dir=cache_dir
                )
        dtype = torch.float16 if fp16 else torch.float32
        sr_model = get_sr_model(device, unet_sr_path, dtype=dtype)
        return KandiSuperResPipeline(scale, device, dtype, flash, sr_model)