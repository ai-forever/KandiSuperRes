import math
import torch
from tqdm import tqdm
from .utils import get_tensor_items
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

        self.time_scale = 1000 // self.num_timesteps
        self.gen_noise = gen_noise
        
    def get_x_start(self, x, t, noise):
        sqrt_one_minus_alphas_cumprod = get_tensor_items(self.sqrt_one_minus_alphas_cumprod, t, noise.shape)
        sqrt_alphas_cumprod = get_tensor_items(self.sqrt_alphas_cumprod, t, noise.shape)
        pred_x_start = (x - sqrt_one_minus_alphas_cumprod * noise) / sqrt_alphas_cumprod
        return pred_x_start
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = self.gen_noise(x_start)
        sqrt_alphas_cumprod = get_tensor_items(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod = get_tensor_items(self.sqrt_one_minus_alphas_cumprod, t, noise.shape)
        x_t = sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise
        return x_t
       
    @torch.no_grad()
    def refine(self, model, img, context, context_mask):
#         for time in tqdm([479, 229]):
        for time in [229]:
            time = torch.tensor([time,] * img.shape[0], device=img.device)
            x_t = self.q_sample(img, time)
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
    
    
    def refine_tiled(self, model, img, context, context_mask):       
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
                tile = self.refine(model, tile, context, context_mask)
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
    

def get_diffusion(conf):
    betas = get_named_beta_schedule(**conf.schedule_params)
    base_diffusion = BaseDiffusion(betas, **conf.diffusion_params)
    return base_diffusion