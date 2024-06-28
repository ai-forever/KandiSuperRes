from typing import Union, List
import PIL
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as T
import torch.nn.functional as F
from einops import repeat

from KandiSuperRes.model.unet import UNet
from KandiSuperRes.model.unet_sr import UNet as UNet_sr
from KandiSuperRes.movq import MoVQ
from KandiSuperRes.model.diffusion import BaseDiffusion, get_named_beta_schedule
# from kandinsky3.model.diffusion_sr import DPMSolver
from KandiSuperRes.model.diffusion_sr_turbo import BaseDiffusion as BaseDiffusion_turbo

import time
from tqdm import tqdm


class Kandinsky3SRPipeline:
    
    def __init__(
        self,
        device_map: Union[str, torch.device, dict],
        dtype_map: Union[str, torch.dtype, dict],
        sr_model: UNet_sr,
        movq: MoVQ,
        refiner: UNet = None,
    ):
        self.device_map = device_map
        self.dtype_map = dtype_map
        self.to_pil = T.ToPILImage()
        self.to_tensor = T.ToTensor()
        
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.ToTensor(),
            T.Lambda(lambda img: 2. * img - 1.),
        ])
        
        self.sr_model = sr_model
        self.movq = movq
        self.refiner = refiner
        
        self.scale = 2
      
    def __call__(
        self, 
        image: PIL.Image.Image,
        images_num: int = 1,
        bs: int = 1, 
        steps: int = 50,
        steps_sr: int = 4,
        eta: float = 0.0, 
        noise=None,
        refine=True
    ) -> List[PIL.Image.Image]:
        
        betas_turbo = get_named_beta_schedule('linear', 1000)
        base_diffusion_sr = BaseDiffusion_turbo(betas_turbo)
        # base_diffusion_sr = DPMSolver(steps_sr)

        old_height = image.size[1]
        old_width = image.size[0]
        height = int(old_height-np.mod(old_height,32))
        width = int(old_width-np.mod(old_width,32))

        image = image.resize((width,height))
        lr_image = self.image_transform(image).unsqueeze(0).to(self.device_map['sr_model'])
        lr_image = lr_image.repeat_interleave(bs, 0)
        
        sr_image = base_diffusion_sr.p_sample_loop(
            self.sr_model, (bs, 3, height*self.scale, width*self.scale), self.device_map['sr_model'], self.dtype_map['sr_model'], lowres_img=lr_image
        )
        
        # sr_image = base_diffusion_sr.generate_panorama(height, width, self.device_map['sr_model'], self.dtype_map['sr_model'], 
        #                                                steps, self.sr_model, lowres_img=lr_image, 
        #                                               view_batch_size=15, eta=eta, seed=0)

        if refine:
            betas = get_named_beta_schedule('cosine', 1000)
            base_diffusion = BaseDiffusion(betas, 0.99)
            
            with torch.cuda.amp.autocast(dtype=self.dtype_map['movq']):
                lr_image_latent = self.movq.encode(sr_image)
            
            pil_images = []
            context = torch.load('../weights/context.pt').to(self.device_map['text_encoder'])
            context_mask = torch.load('../weights/context_mask.pt').to(self.device_map['text_encoder'])
            
            with torch.no_grad():
                bs_context = repeat(context, '1 n d -> b n d', b=bs)
                bs_context_mask = repeat(context_mask, '1 n -> b n', b=bs)
    
                with torch.cuda.amp.autocast(dtype=self.dtype_map['refiner']):
                    refiner_images = base_diffusion.refine_tiled(self.refiner, lr_image_latent, bs_context, bs_context_mask, noise)
                    
                with torch.cuda.amp.autocast(dtype=self.dtype_map['movq']):
                    refiner_image = self.movq.decode(refiner_images)
                    refiner_image = torch.clip((refiner_image + 1.) / 2., 0., 1.)
                
            if old_height*self.scale != refiner_image.shape[2] or old_width*self.scale != refiner_image.shape[3]:
                refiner_image = F.interpolate(refiner_image, [old_height*self.scale, old_width*self.scale], mode='bilinear', align_corners=True)
            refined_pil_image = self.to_pil(refiner_image[0])
            return refined_pil_image

        sr_image = torch.clip((sr_image + 1.) / 2., 0., 1.)
        if old_height*self.scale != sr_image.shape[2] or old_width*self.scale != sr_image.shape[3]:
            sr_image = F.interpolate(sr_image, [old_height*self.scale, old_width*self.scale], mode='bilinear', align_corners=True)
        pil_sr_image = self.to_pil(sr_image[0])
        return pil_sr_image