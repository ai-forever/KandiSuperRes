import math

import torch
from einops import rearrange
from tqdm import tqdm

from .utils import get_tensor_items
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
import copy
import torch.nn.functional as F


def get_named_beta_schedule(schedule_name, timesteps):
    if schedule_name == "linear":
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start, beta_end, timesteps, dtype=torch.float32
        )
    elif schedule_name == "cosine":
        alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = []
        for i in range(timesteps):
            t1 = i / timesteps
            t2 = (i + 1) / timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        return torch.tensor(betas, dtype=torch.float32)


class BaseDiffusion:

    def __init__(self, betas, percentile=None, gen_noise=torch.randn_like):
        self.betas = betas
        self.num_timesteps = betas.shape[0]

        alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=betas.dtype), self.alphas_cumprod[:-1]])

        # calculate q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculate q(x_{t-1} | x_t, x_0)
        self.posterior_mean_coef_1 = torch.sqrt(self.alphas_cumprod_prev) * betas / (1. - self.alphas_cumprod)
        self.posterior_mean_coef_2 = torch.sqrt(alphas) * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )

        self.percentile = percentile
        self.time_scale = 1000 // self.num_timesteps
        self.gen_noise = gen_noise
        self.jump_length = 3
        
        self.dpm_solver = DPMSolverMultistepScheduler(
            beta_schedule='squaredcos_cap_v2',#"linear",
#             prediction_type= "sample",
            algorithm_type="sde-dpmsolver++",
            dynamic_thresholding_ratio=0.995, 
            thresholding=True
        )
        print(self.num_timesteps)
        self.dpm_solver.set_timesteps(self.num_timesteps)
        

    def process_x_start(self, x_start):
        bs, ndims = x_start.shape[0], len(x_start.shape[1:])
        if self.percentile is not None:
            quantile = torch.quantile(
                rearrange(x_start, 'b ... -> b (...)').abs(),
                self.percentile,
                dim=-1
            )
            quantile = torch.clip(quantile, min=1.)
            quantile = quantile.reshape(bs, *((1,) * ndims))
            return torch.clip(x_start, -quantile, quantile) / quantile
        else:
            return torch.clip(x_start, -1., 1.)
        
    def get_x_start(self, x, t, noise):
        sqrt_one_minus_alphas_cumprod = get_tensor_items(self.sqrt_one_minus_alphas_cumprod, t, noise.shape)
        sqrt_alphas_cumprod = get_tensor_items(self.sqrt_alphas_cumprod, t, noise.shape)
        pred_x_start = (x - sqrt_one_minus_alphas_cumprod * noise) / sqrt_alphas_cumprod
        return pred_x_start
    
    def get_noise(self, x, t, x_start):
        sqrt_one_minus_alphas_cumprod = get_tensor_items(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        sqrt_alphas_cumprod = get_tensor_items(self.sqrt_alphas_cumprod, t, x_start.shape)
        pred_noise = (x - sqrt_alphas_cumprod * x_start) / sqrt_one_minus_alphas_cumprod
        return pred_noise

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = self.gen_noise(x_start)
        sqrt_alphas_cumprod = get_tensor_items(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod = get_tensor_items(self.sqrt_one_minus_alphas_cumprod, t, noise.shape)
        x_t = sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise
        return x_t

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean_coef_1 = get_tensor_items(self.posterior_mean_coef_1, t, x_start.shape)
        posterior_mean_coef_2 = get_tensor_items(self.posterior_mean_coef_2, t, x_t.shape)
        posterior_mean = posterior_mean_coef_1 * x_start + posterior_mean_coef_2 * x_t

        posterior_variance = get_tensor_items(self.posterior_variance, t, x_start.shape)
        posterior_log_variance = get_tensor_items(self.posterior_log_variance, t, x_start.shape)
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def q_posterior_variance(self, t, prev_t, shape, eta=1.,):
        alphas_cumprod = get_tensor_items(self.alphas_cumprod, t, shape)
        prev_alphas_cumprod = get_tensor_items(self.alphas_cumprod, prev_t, shape)

        posterior_variance = torch.sqrt(
            eta * (1. - alphas_cumprod / prev_alphas_cumprod) * (1. - prev_alphas_cumprod) / (1. - alphas_cumprod)
        )
        return posterior_variance

    def text_guidance(
        self, model, x, t, context, context_mask, null_embedding, guidance_weight_text, 
        uncondition_context=None, uncondition_context_mask=None, mask=None, masked_latent=None
    ):
        bs = x.shape[0]
        large_x = x.repeat(2, 1, 1, 1)
        large_t = t.repeat(2).to(x.dtype)
        
        is_text = torch.tensor([True for _ in range(bs)], dtype=torch.bool)
        uncondition_is_text = torch.tensor([True for _ in range(bs)], dtype=torch.bool)
        if uncondition_context is None:
            uncondition_context = torch.zeros_like(context)
            uncondition_context_mask = torch.zeros_like(context_mask)
            uncondition_context[:, 0] = null_embedding
            uncondition_context_mask[:, 0] = 1
            uncondition_is_text = ~uncondition_is_text
        large_context = torch.cat([context, uncondition_context])
        large_context_mask = torch.cat([context_mask, uncondition_context_mask])
        large_is_text = torch.cat([is_text, uncondition_is_text])
        
        if mask is not None:
            mask = mask.repeat(2, 1, 1, 1)
        if masked_latent is not None:
            masked_latent = masked_latent.repeat(2, 1, 1, 1)
        
        if model.in_layer.in_channels == 9:
            large_x = torch.cat([large_x, mask, masked_latent], dim=1)
        
        pred_large_noise = model(
            large_x, large_t * self.time_scale, large_context, large_context_mask.bool(), large_is_text, null_embedding
        )
        pred_noise, uncond_pred_noise = torch.chunk(pred_large_noise, 2)
        pred_noise = (guidance_weight_text + 1.) * pred_noise - guidance_weight_text * uncond_pred_noise
        return pred_noise

    def p_mean_variance(
        self, model, x, t, prev_t, context, context_mask, null_embedding, guidance_weight_text, eta=1.,
        negative_context=None, negative_context_mask=None, mask=None, masked_latent=None
    ):
        
        pred_noise = self.text_guidance(
            model, x, t, context, context_mask, null_embedding, guidance_weight_text,
            negative_context, negative_context_mask, mask, masked_latent
        )

        pred_x_start = self.get_x_start(x, t, pred_noise)
        pred_x_start = self.process_x_start(pred_x_start)
        pred_noise = self.get_noise(x, t, pred_x_start)
        pred_var = self.q_posterior_variance(t, prev_t, x.shape, eta)
        
        prev_alphas_cumprod = get_tensor_items(self.alphas_cumprod, prev_t, x.shape)
        pred_mean = torch.sqrt(prev_alphas_cumprod) * pred_x_start
        pred_mean += torch.sqrt(1. - prev_alphas_cumprod - pred_var**2) * pred_noise
        return pred_mean, pred_var

    @torch.no_grad()
    def p_sample(
        self, model, x, t, prev_t, context, context_mask, null_embedding, guidance_weight_text, eta=1.,
        negative_context=None, negative_context_mask=None, mask=None, masked_latent=None
    ):
        bs = x.shape[0]
        ndims = len(x.shape[1:])
        pred_mean, pred_var = self.p_mean_variance(
            model, x, t, prev_t, context, context_mask, null_embedding, guidance_weight_text, eta,
            negative_context=negative_context, negative_context_mask=negative_context_mask,
            mask=mask, masked_latent=masked_latent
        )
        noise = torch.randn_like(x)
        mask = (prev_t != 0).reshape(bs, *((1,) * ndims))
        sample = pred_mean + mask * pred_var * noise
        return sample

    @torch.no_grad()
    def p_sample_loop(
        self, model, shape, times, device, context, context_mask, null_embedding, guidance_weight_text, eta=1.,
        negative_context=None, negative_context_mask=None, mask=None, masked_latent=None, gan=False,
    ):
        img = torch.randn(*shape, device=device)
        times = times + [0,]
        times = list(zip(times[:-1], times[1:]))
        
        for time, prev_time in tqdm(times):
            time = torch.tensor([time] * shape[0], device=device)
            if gan:
                x_t = self.q_sample(img, time)
                pred_noise = model(x_t, time.type(x_t.dtype), context, context_mask.bool())
                img = self.get_x_start(x_t, time, pred_noise)
            else:
                prev_time = torch.tensor([prev_time] * shape[0], device=device)
                img = self.p_sample(
                    model, img, time, prev_time, context, context_mask, null_embedding, guidance_weight_text, eta,
                    negative_context=negative_context, negative_context_mask=negative_context_mask,
                    mask=mask, masked_latent=masked_latent
                )
        return img
    
    
    @torch.no_grad()
    def refine(self, model, img, context, context_mask, noise):
#         for time in tqdm([479, 229]):
        for time in [229]:
            time = torch.tensor([time,] * img.shape[0], device=img.device)
            x_t = self.q_sample(img, time, noise=noise)
            pred_noise = model(x_t, time.type(x_t.dtype), context, context_mask.bool())
            img = self.get_x_start(x_t, time, pred_noise)
        return img
    
    def blend_v(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[ :, :, y, :] = a[ :, :, -blend_extent + y, :] * (
                1 - y / blend_extent
            ) + b[ :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[ :, :, :, x] = a[ :, :, :, -blend_extent + x] * (
                1 - x / blend_extent
            ) + b[ :, :, :, x] * (x / blend_extent)
        return b
    
    
    @torch.no_grad()
    def refine_tiled2(self, model, img, context, context_mask):       
        tile_sample_min_size = 352
        tile_overlap_factor = 0.25
            
        overlap_size = int(tile_sample_min_size * (1 - tile_overlap_factor))
        tile_latent_min_size = int(tile_sample_min_size)
        blend_extent = int(tile_latent_min_size * tile_overlap_factor)
        row_limit = tile_latent_min_size - blend_extent
        
        for time in tqdm([479, 229]):
            # Split the image into tiles and encode them separately.
            rows = []
            for i in range(0, img.shape[2], overlap_size):
                row = []
                for j in range(0, img.shape[3], overlap_size):
                    tile = img[
                        :,
                        :,
                        i : i + tile_sample_min_size,
                        j : j + tile_sample_min_size,
                    ]
#                     print(tile.shape)
                    
                    time = torch.tensor([time,] * tile.shape[0], device=tile.device)
                    x_t = self.q_sample(tile, time)
                    pred_noise = model(x_t, time.type(x_t.dtype), context, context_mask.bool())
                    refined_tile = self.get_x_start(x_t, time, pred_noise)
            
#                     tile = self.refine(model, tile, context, context_mask)
                    row.append(refined_tile)
                rows.append(row)
            result_rows = []
            for i, row in enumerate(rows):
                result_row = []
                for j, tile in enumerate(row):
                    # blend the above tile and the left tile
                    # to the current tile and add the current tile to the result row
                    if i > 0:
                        tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                    if j > 0:
                        tile = self.blend_h(row[j - 1], tile, blend_extent)
                    result_row.append(tile[ :, :, :row_limit, :row_limit])
                result_rows.append(torch.cat(result_row, dim=3))

            img = torch.cat(result_rows, dim=2)
            
        return img
    
    
    def refine_tiled(self, model, img, context, context_mask, noise):       
        tile_sample_min_size = 352
        tile_overlap_factor = 0.25
            
        overlap_size = int(tile_sample_min_size * (1 - tile_overlap_factor))
        tile_latent_min_size = int(tile_sample_min_size)
        blend_extent = int(tile_latent_min_size * tile_overlap_factor)
        row_limit = tile_latent_min_size - blend_extent
        
        # Split the image into tiles and encode them separately.
        rows = []
        for i in tqdm(range(0, img.shape[2], overlap_size)):
            row = []
            for j in range(0, img.shape[3], overlap_size):
                tile = img[
                    :,
                    :,
                    i : i + tile_sample_min_size,
                    j : j + tile_sample_min_size,
                ]
                tile = self.refine(model, tile, context, context_mask, noise)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[ :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        refine_img = torch.cat(result_rows, dim=2)
        return refine_img
    
    
    def get_views(self, panorama_height, panorama_width, window_size=352, stride=256):
        # Here, we define the mappings F_i (see Eq. 7 in the MultiDiffusion paper https://arxiv.org/abs/2302.08113)
        # if panorama's height/width < window_size, num_blocks of height/width should return 1
        num_blocks_height = round(math.ceil((panorama_height - window_size) / stride)) + 1 if panorama_height > window_size else 1
        num_blocks_width = round(math.ceil((panorama_width - window_size) / stride)) + 1 if panorama_width > window_size else 1
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        print(total_num_blocks)
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
    
    @torch.no_grad()
    def refine_panorama(self, model, img, context, context_mask, view_batch_size=1):
        # 6. Define panorama grid and initialize views for synthesis.
        # prepare batch grid
        height, width = img.shape[2], img.shape[3] 
        views = self.get_views(height, width)
        views_batch = [views[i : i + view_batch_size] for i in range(0, len(views), view_batch_size)]
        views_scheduler_status = [copy.deepcopy(self.dpm_solver.__dict__)] * len(views_batch)

        shape = (1, 4, height, width)
        count = torch.zeros(*shape, device=img.device)
        value = torch.zeros(*shape, device=img.device)
        
#         generator = torch.Generator(device=img.device)
#         if seed is not None:
#             generator = generator.manual_seed(seed)
            
#         img = torch.randn(*shape, device=device, generator=generator)
#         up_lowres_img = F.interpolate(lowres_img, (shape[2], shape[3]), mode="bilinear")

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
#         extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # 8. Denoising loop
        # Each denoising step also includes refinement of the latents with respect to the
        # views.
#         timesteps = self.dpm_solver.timesteps
#         num_warmup_steps = len(timesteps) - num_inference_steps * self.dpm_solver.order
        
#         for time in tqdm([2, 1]):
        for time in tqdm([479, 229]):
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

#                 # rematch block's scheduler status
#                 self.dpm_solver.__dict__.update(views_scheduler_status[j])

                time = torch.tensor([time,] * img_for_view.shape[0], device=img.device)
                x_t = self.q_sample(img_for_view, time)
                pred_noise = model(x_t, time.type(x_t.dtype), context, context_mask.bool())
                img_denoised_batch = self.get_x_start(x_t, time, pred_noise)
                
#                 t = torch.tensor([time] * img_for_view.shape[0], device=device)
#                 pred_noise = self.pred_noise(
#                     unet, img_for_view, t, lowres_img_for_view, dtype
#                 )
#                 img_denoised_batch = self.dpm_solver.step(pred_noise, time, img_for_view).prev_sample
#                 img_denoised_batch = self.dpm_solver.step(pred_noise, time, img_for_view, **extra_step_kwargs).prev_sample
                # save views scheduler status after sample
#                 views_scheduler_status[j] = copy.deepcopy(self.dpm_solver.__dict__)

                # extract value from batch
                for img_view_denoised, (h_start, h_end, w_start, w_end) in zip(
                    img_denoised_batch.chunk(vb_size), batch_view
                ):
                    value[:, :, h_start:h_end, w_start:w_end] += img_view_denoised
                    count[:, :, h_start:h_end, w_start:w_end] += 1

            # take the MultiDiffusion step. Eq. 5 in MultiDiffusion paper: https://arxiv.org/abs/2302.08113
            img = torch.where(count > 0, value / count, value)
        print(img.shape)    
        return img

def get_diffusion(conf):
    betas = get_named_beta_schedule(**conf.schedule_params)
    base_diffusion = BaseDiffusion(betas, **conf.diffusion_params)
    return base_diffusion
