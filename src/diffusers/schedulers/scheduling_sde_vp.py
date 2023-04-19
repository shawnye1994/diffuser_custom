# Copyright 2023 Google Brain and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DISCLAIMER: This file is strongly influenced by https://github.com/yang-song/score_sde_pytorch

import math
from typing import Optional, Tuple, Union

import torch
from dataclasses import dataclass
from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, randn_tensor
from .scheduling_utils import SchedulerMixin
import jax.numpy as jnp
import jax
import numpy as np

@dataclass
class SdeVpOutput(BaseOutput):
    """
    Output class for the ScoreSdeVpScheduler's step function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor

class ScoreSdeVpScheduler(SchedulerMixin, ConfigMixin):
    """
    The variance preserving stochastic differential equation (SDE) scheduler.

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    For more information, see the original paper: https://arxiv.org/abs/2011.13456

    UNDER CONSTRUCTION

    """

    order = 1

    @register_to_config
    def __init__(self, num_train_timesteps=1000, beta_min=0.1, beta_max=20, likelihood_weighting=True, importance_weighting=True, eps=1e-5):
        """
        beta_min = 0.1, beta_max=20 corresponds to the default DDPM of (beta_start=1e-4, beta_end=20)
        """
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.likelihood_weighting = likelihood_weighting
        self.importance_weighting = importance_weighting

        self.num_train_timesteps = num_train_timesteps
        self.N = num_train_timesteps
        self.discrete_betas = torch.linspace(beta_min / self.N, beta_max / self.N, self.N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.timesteps = None
        self.rng = jax.random.PRNGKey(2022)
        self.eps = eps
    
    @property
    def T(self):
        return 1.
    

    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        return sample
    
    def _sde(self, x, t):
        """
        The sde (forward diffusion process) corresponds to the DDPM model by Ho. et al. 2020
        beta(t) = beta_min + t*(beta_max - beta_min), t\in[0,1]
        dxt = -0.5*beta(t)*xt*dt + sqrt(beta(t))*dwt
        Args:
            x: (N, C, H, W)
            t: (N, ), \in[0,1]
        Returns:
            drift: (N, C, H, W)
            diffusion: (N, )
        """
        beta_t = self.beta_0 + t*(self.beta_1 - self.beta_0)
        drift = -0.5 * x * beta_t[:, None, None, None]
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion
    
    def sample_timesteps(self, batch_size, device):
        """
        Return: t: (N, ), torch tensor on device
        """
        if self.likelihood_weighting and self.importance_weighting:
            t = self._sample_importance_weighted_time_for_likelihood((batch_size,), eps=self.eps)
            t = np.asarray(t)
            t = torch.from_numpy(t).to(device)
        else:
            t = torch.rand(N, device=device) * (self.T - self.eps) + self.eps
        
        return t*999

    def _marginal_prob(self, x0, t):
        """
        Calculate the mean and std of the marginal distribution q(xt|x0), can be used for sampling xt
        q(xt|x0) = N(xt;gamma_t*x0, var_t*I)
        int_beta_t = int_0^t{beta(s)ds}
        mean_t = exp(-0.5*int_beta_t)
        var_t = 1 - exp(-int_beta_t)
        Args:
            x0: (N, C, H, W)
            t: (N, ), \in[0,1]
        Returns:
            mean_t: (N, C, H, W)
            std_t: (N, )
        """
        int_beta_t = t*self.beta_0 + 0.5*torch.square(t)*(self.beta_max - self.beta_1)
        mean_t = torch.exp(-0.5 * int_beta_t[:, None, None, None]) * x0
        std_t = torch.sqrt(1 - torch.exp(-int_beta_t))
        return mean_t, std_t
    
    def set_timesteps(self, num_inference_steps, device: Union[str, torch.device] = None):
        self.timesteps = torch.linspace(self.T, self.eps, self.N, device=self.device)
    
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[SdeVpOutput, Tuple]:
        """
        dt = -1./self.N

        eps = torch.randn_like(x)
        t = self.timesteps[timestep]
        drift, diffusion = self.rsde.sde(sample, torch.ones(sample.shape[0], device = sample.device)*t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * eps
        """
        #To Do
        return SdeVpOutput(prev_sample=prev_sample)

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.FloatTensor):
        
        mean_t, std_t = self._marginal_prob(original_samples, timesteps)
        noisy_samples = mean_t + noise * std_t[:, None, None, None]

        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
    
    def _likelihood_importance_cum_weight(self, t, eps=1e-5):
        exponent1 = 0.5 * eps * (eps * self.beta_1 - (eps - 2) * self.beta_0)
        exponent2 = 0.5 * t * (self.beta_1 * t - (t - 2) * self.beta_0)
        term1 = jnp.where(exponent1 <= 1e-3, jnp.log(exponent1), jnp.log(jnp.exp(exponent1) - 1.))
        term2 = jnp.where(exponent2 <= 1e-3, jnp.log(exponent2), jnp.log(jnp.exp(exponent2) - 1.))
        return 0.5 * (-4 * term1 + 4 * term2
                    + (2 * eps - eps ** 2 + t * (t - 2)) * self.beta_0 + (eps ** 2 - t ** 2) * self.beta_1)

    def _sample_importance_weighted_time_for_likelihood(self, shape, quantile=None, eps=1e-5, steps=100):
        Z = self._likelihood_importance_cum_weight(self.T, eps=eps)
        if quantile is None:
            _, self.rng = jax.random.split(self.rng)
            quantile = jax.random.uniform(self.rng, shape, minval=0, maxval=Z)
        lb = jnp.ones_like(quantile) * eps
        ub = jnp.ones_like(quantile) * self.T

        def bisection_func(carry, idx):
            lb, ub = carry
            mid = (lb + ub) / 2.
            value = self._likelihood_importance_cum_weight(mid, eps=eps)
            lb = jnp.where(value <= quantile, mid, lb)
            ub = jnp.where(value <= quantile, ub, mid)
            return (lb, ub), idx

        (lb, ub), _ = jax.lax.scan(bisection_func, (lb, ub), jnp.arange(0, steps))
        return (lb + ub) / 2.