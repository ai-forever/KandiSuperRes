from superres.model.diffusion import DPMSolver
import PIL
import torchvision.transforms as T
import torch.nn.functional as F
from superres.model.unet import UNet
import torch
import numpy as np


class SuperResPipeline:
    
    def __init__(
        self, 
        device: str,
        dtype: str,
        unet: UNet,   
    ):
        self.device = device
        self.dtype = dtype
        self.scale = 4
        
        self.to_pil = T.ToPILImage()
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.ToTensor(),
            T.Lambda(lambda img: 2. * img - 1.),
        ])
        
        self.unet = unet
        
    def __call__(
        self, 
        pil_image: PIL.Image.Image = None,
        steps: int = 5
    ) -> PIL.Image.Image:
        
        base_diffusion = DPMSolver(steps)
        
        lr_image = self.image_transform(pil_image).unsqueeze(0).to(self.device)
        
        old_height = pil_image.size[1]
        old_width = pil_image.size[0]

        height = int(old_height+np.mod(old_height,2))*self.scale
        width = int(old_width+np.mod(old_width,2))*self.scale

        sr_image = base_diffusion.generate_panorama(height, width, self.device, steps, 
                                                   self.unet, lowres_img=lr_image.to(dtype=self.dtype), 
                                                   view_batch_size=15, eta=0.0, seed=0)

        sr_image = torch.clip((sr_image + 1.) / 2., 0., 1.)
        if old_height*4 != height or old_width*4 != width:
            sr_image = F.interpolate(sr_image, [old_height*4, old_width*4], mode='bilinear', align_corners=True)
            
        pil_sr_image = self.to_pil(sr_image[0])
        return pil_sr_image