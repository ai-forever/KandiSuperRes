from KandiSuperRes.model.diffusion import DPMSolver
import PIL
import torchvision.transforms as T
import torch.nn.functional as F
from KandiSuperRes.model.unet import UNet
import torch
import numpy as np


class KandiSuperResPipeline:
    
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
        steps: int = 5,
        view_batch_size: int = 15,
        seed: int = 0
    ) -> PIL.Image.Image:
        
        base_diffusion = DPMSolver(steps)
        
        lr_image = self.image_transform(pil_image).unsqueeze(0).to(self.device)
        
        old_height = pil_image.size[1]
        old_width = pil_image.size[0]

        height = int(old_height+np.mod(old_height,2))*self.scale
        width = int(old_width+np.mod(old_width,2))*self.scale

        sr_image = base_diffusion.generate_panorama(height, width, self.device, self.dtype, steps, 
                                                   self.unet, lowres_img=lr_image, 
                                                   view_batch_size=view_batch_size, eta=0.0, seed=seed)

        sr_image = torch.clip((sr_image + 1.) / 2., 0., 1.)
        if old_height*self.scale != height or old_width*self.scale != width:
            sr_image = F.interpolate(sr_image, [old_height*self.scale, old_width*self.scale], mode='bilinear', align_corners=True)
            
        pil_sr_image = self.to_pil(sr_image[0])
        return pil_sr_image
