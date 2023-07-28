import torch

from diffusionModels.simpleDiffusion.varianceSchedule import VarianceSchedule

from utils.networkHelper import *

import torch.nn.functional as F


class DiffusionModel(nn.Module):
    def __init__(self, schedule_name='linear_beta_schedule', timesteps=1000, beta_start=0.0001, beta_end=0.02,
                 denoise_model=None):
        super(DiffusionModel, self).__init__()
        self.denoise_model = denoise_model

        # 方差产生
        variance_schedule_func = VarianceSchedule(schedule_name, beta_start, beta_end)
        self.timesteps = timesteps
        self.betas = variance_schedule_func(timesteps)

        # define alphas
        self.alphas = 1. - self.betas  # alpha
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # 累乘的 alpha  累乘函数 [1,2,3,4,5] -> [1,2,6,24,120]
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)  # 填充 左填充 1个值 右填充 0个值
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1} ) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt((1. - self.alphas_cumprod))

        # calculations for posterior q(x_{t-1} | x_t, x_0 )
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, x_start, t, noise=None, loss_type='l1'):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.denoise_model(x_noisy, t)
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == 'huber':
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)