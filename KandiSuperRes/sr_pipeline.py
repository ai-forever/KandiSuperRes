import torch
import numpy as np
import PIL
import torchvision.transforms as T
import torch.nn.functional as F
from KandiSuperRes.model.unet import UNet
from KandiSuperRes.model.unet_sr import UNet as UNet_sr
from KandiSuperRes.movq import MoVQ
from KandiSuperRes.model.diffusion_sr import DPMSolver
from KandiSuperRes.model.diffusion_refine import BaseDiffusion, get_named_beta_schedule
from KandiSuperRes.model.diffusion_sr_turbo import BaseDiffusion as BaseDiffusion_turbo


class KandiSuperResPipeline:
    
    def __init__(
        self, 
        scale: int,
        device: str,
        dtype: str,
        flash: bool,
        sr_model: UNet_sr,
        movq: MoVQ = None,
        refiner: UNet = None,
    ):
        self.device = device
        self.dtype = dtype
        self.scale = scale
        self.flash = flash
        self.to_pil = T.ToPILImage()
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.ToTensor(),
            T.Lambda(lambda img: 2. * img - 1.),
        ])
        
        self.sr_model = sr_model
        self.movq = movq
        self.refiner = refiner
        
    def __call__(
        self, 
        pil_image: PIL.Image.Image = None,
        steps: int = 5,
        view_batch_size: int = 15,
        seed: int = 0,
        refine=True
    ) -> PIL.Image.Image:

        if self.flash:
            betas_turbo = get_named_beta_schedule('linear', 1000)
            base_diffusion_sr = BaseDiffusion_turbo(betas_turbo)
    
            old_height = pil_image.size[1]
            old_width = pil_image.size[0]
            height = int(old_height-np.mod(old_height,32))
            width = int(old_width-np.mod(old_width,32))
    
            pil_image = pil_image.resize((width,height))
            lr_image = self.image_transform(pil_image).unsqueeze(0).to(self.device['sr_model'])
            
            sr_image = base_diffusion_sr.p_sample_loop(
                self.sr_model, (1, 3, height*self.scale, width*self.scale), self.device['sr_model'], self.dtype['sr_model'], lowres_img=lr_image
            )

            if refine:
                betas = get_named_beta_schedule('cosine', 1000)
                base_diffusion = BaseDiffusion(betas, 0.99)
                
                with torch.cuda.amp.autocast(dtype=self.dtype['movq']):
                    lr_image_latent = self.movq.encode(sr_image)
                
                pil_images = []
                context = torch.load('weights/context.pt').to(self.dtype['refiner'])
                context_mask = torch.load('weights/context_mask.pt').to(self.dtype['refiner'])
                
                with torch.no_grad():       
                    with torch.cuda.amp.autocast(dtype=self.dtype['refiner']):
                        refiner_image = base_diffusion.refine_tiled(self.refiner, lr_image_latent, context, context_mask)
                        
                    with torch.cuda.amp.autocast(dtype=self.dtype['movq']):
                        refiner_image = self.movq.decode(refiner_image)
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

        else:
            base_diffusion = DPMSolver(steps)
            
            lr_image = self.image_transform(pil_image).unsqueeze(0).to(self.device)
            
            old_height = pil_image.size[1]
            old_width = pil_image.size[0]
    
            height = int(old_height+np.mod(old_height,2))*self.scale
            width = int(old_width+np.mod(old_width,2))*self.scale
    
            sr_image = base_diffusion.generate_panorama(height, width, self.device, self.dtype, steps, 
                                                       self.sr_model, lowres_img=lr_image, 
                                                       view_batch_size=view_batch_size, eta=0.0, seed=seed)
    
            sr_image = torch.clip((sr_image + 1.) / 2., 0., 1.)
            if old_height*self.scale != height or old_width*self.scale != width:
                sr_image = F.interpolate(sr_image, [old_height*self.scale, old_width*self.scale], mode='bilinear', align_corners=True)
                
            pil_sr_image = self.to_pil(sr_image[0])
            return pil_sr_image
