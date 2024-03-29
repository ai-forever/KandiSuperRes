from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
from einops import repeat
import copy
import inspect
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm


class DPMSolver:
    
    def __init__(self, num_timesteps):
        self.dpm_solver = DPMSolverMultistepScheduler(
            beta_schedule="linear",
            prediction_type= "sample",
#             algorithm_type="sde-dpmsolver++",
            thresholding=False
        )
        self.dpm_solver.set_timesteps(num_timesteps)
        
    @torch.no_grad()    
    def pred_noise(self, model, x, t, lowres_img, dtype):
        pred_noise = model(x.to(dtype), t.to(dtype), lowres_img=lowres_img.to(dtype))
        pred_noise = pred_noise.to(dtype=torch.float32)
        return pred_noise
    
    
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.dpm_solver.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.dpm_solver.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    
    def get_views(self, panorama_height, panorama_width, window_size=1024, stride=800):
        # Here, we define the mappings F_i (see Eq. 7 in the MultiDiffusion paper https://arxiv.org/abs/2302.08113)
        # if panorama's height/width < window_size, num_blocks of height/width should return 1
        num_blocks_height = round(math.ceil((panorama_height - window_size) / stride)) + 1 if panorama_height > window_size else 1
        num_blocks_width = round(math.ceil((panorama_width - window_size) / stride)) + 1 if panorama_width > window_size else 1
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        views = []
        for i in range(total_num_blocks):
            h_start = int((i // num_blocks_width) * stride)
            h_end = h_start + window_size
            if h_end > panorama_height and num_blocks_height > 1:
                h_end = panorama_height
                h_start = panorama_height - window_size
            w_start = int((i % num_blocks_width) * stride)
            w_end = w_start + window_size
            if w_end > panorama_width and num_blocks_width > 1:
                w_end = panorama_width
                w_start = panorama_width - window_size
            views.append((h_start, h_end, w_start, w_end))
        return views
    
    
    def generate_panorama(self, height, width, device, dtype, num_inference_steps, 
                  unet, lowres_img, view_batch_size=15, eta=0, seed=0):
        # 6. Define panorama grid and initialize views for synthesis.
        # prepare batch grid
        views = self.get_views(height, width)
        views_batch = [views[i : i + view_batch_size] for i in range(0, len(views), view_batch_size)]
        views_scheduler_status = [copy.deepcopy(self.dpm_solver.__dict__)] * len(views_batch)

        shape = (1, 3, height, width)
        count = torch.zeros(*shape, device=device)
        value = torch.zeros(*shape, device=device)
        
        generator = torch.Generator(device=device)
        if seed is not None:
            generator = generator.manual_seed(seed)
            
        img = torch.randn(*shape, device=device, generator=generator)
        up_lowres_img = F.interpolate(lowres_img, (shape[2], shape[3]), mode="bilinear")

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # 8. Denoising loop
        # Each denoising step also includes refinement of the latents with respect to the
        # views.
        timesteps = self.dpm_solver.timesteps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.dpm_solver.order
        
        for i, time in tqdm(enumerate(self.dpm_solver.timesteps)):
            count.zero_()
            value.zero_()
            
            # generate views
            # Here, we iterate through different spatial crops of the latents and denoise them. These
            # denoised (latent) crops are then averaged to produce the final latent
            # for the current timestep via MultiDiffusion. Please see Sec. 4.1 in the
            # MultiDiffusion paper for more details: https://arxiv.org/abs/2302.08113
            # Batch views denoise
            for j, batch_view in enumerate(views_batch):
                vb_size = len(batch_view)
                # get the latents corresponding to the current view coordinates  
                img_for_view = torch.cat(
                    [
                        img[:, :, h_start:h_end, w_start:w_end]
                        for h_start, h_end, w_start, w_end in batch_view
                    ]
                )
                lowres_img_for_view = torch.cat(
                    [
                        up_lowres_img[:, :, h_start:h_end, w_start:w_end]
                        for h_start, h_end, w_start, w_end in batch_view
                    ]
                )

                # rematch block's scheduler status
                self.dpm_solver.__dict__.update(views_scheduler_status[j])
                
                t = torch.tensor([time] * img_for_view.shape[0], device=device)
                pred_noise = self.pred_noise(
                    unet, img_for_view, t, lowres_img_for_view, dtype
                )
                img_denoised_batch = self.dpm_solver.step(pred_noise, time, img_for_view, **extra_step_kwargs).prev_sample
                
                # save views scheduler status after sample
                views_scheduler_status[j] = copy.deepcopy(self.dpm_solver.__dict__)

                # extract value from batch
                for img_view_denoised, (h_start, h_end, w_start, w_end) in zip(
                    img_denoised_batch.chunk(vb_size), batch_view
                ):
                    value[:, :, h_start:h_end, w_start:w_end] += img_view_denoised
                    count[:, :, h_start:h_end, w_start:w_end] += 1

            # take the MultiDiffusion step. Eq. 5 in MultiDiffusion paper: https://arxiv.org/abs/2302.08113
            img = torch.where(count > 0, value / count, value)

        return img