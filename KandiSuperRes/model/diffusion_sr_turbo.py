import math

import torch
from einops import rearrange
from tqdm import tqdm

from .utils import get_tensor_items, exist
import numpy as np


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
        self.posterior_mean_coef_1 = (torch.sqrt(self.alphas_cumprod_prev) * betas / (1. - self.alphas_cumprod))
        self.posterior_mean_coef_2 = (torch.sqrt(alphas) * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance = (torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        ))

        self.percentile = percentile
        self.time_scale = 1000 // self.num_timesteps
        self.gen_noise = gen_noise

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = self.gen_noise(x_start)
        sqrt_alphas_cumprod = get_tensor_items(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod = get_tensor_items(self.sqrt_one_minus_alphas_cumprod, t, noise.shape)
        x_t = sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise
        return x_t
    
    @torch.no_grad()
    def p_sample_loop(
        self, model, shape, device,  dtype, lowres_img, times=[979, 729, 479, 229]
    ):
        img = torch.randn(*shape, device=device).to(dtype=dtype)
        times = times + [0,]
        times = list(zip(times[:-1], times[1:]))
        for time, prev_time in tqdm(times):
            time = torch.tensor([time] * shape[0], device=device)
            x_t = self.q_sample(img, time)
            img = model(x_t.to(dtype), time.to(dtype), lowres_img=lowres_img.to(dtype))
        return img
    
    @torch.no_grad()
    def refine(self, model, img, **large_model_kwargs):
        for time in tqdm([729, 479, 229]):
            time = torch.tensor([time,] * img.shape[0], device=img.device)
            x_t = self.q_sample(img, time)
            img = model(x_t, time.type(x_t.dtype), **large_model_kwargs)
        return img

def get_diffusion(conf):
    betas = get_named_beta_schedule(**conf.schedule_params)
    base_diffusion = BaseDiffusion(betas, **conf.diffusion_params)
    return base_diffusion