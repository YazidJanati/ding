from tqdm import tqdm
from typing import Tuple

import torch
from torch import Tensor
from fm.wrappers import NetCFM


def bridge_moments(
    x_0: Tensor, x_1: Tensor, t, t_prev, dm: NetCFM, eta: float = 1.0
) -> Tuple[Tensor, Tensor]:
    """
    Diffusion Model defined as::

        X_t = alpha_t * X_0 + sigma_t * Z

    The same function as ``bridge_kernel`` except it takes as input ``x_0`` and ``x_1``.

    Returns
    -------
    mean, var : (Tensor, Tensor)
        mean and variance of the bridge
    """
    dtype = dm.dtype
    alpha_t, alpha_t_prev = dm.alphas_f64[t], dm.alphas_f64[t_prev]
    sigma_t, sigma_t_prev = dm.sigmas_f64[t], dm.sigmas_f64[t_prev]
    x0_64, x1_64 = x_0.to(torch.float64), x_1.to(torch.float64)

    a_t_t_prev = alpha_t / alpha_t_prev
    sig_t_prev_t = sigma_t_prev / sigma_t

    # NOTE the variance of the DDPM bridge
    var = sigma_t_prev**2 * (1 - (a_t_t_prev * sig_t_prev_t) ** 2)
    var = eta**2 * var

    coef_x_1 = (sigma_t_prev**2 - var).sqrt()
    coef_x_0 = alpha_t_prev

    # cast into the dm dtype
    mean64 = coef_x_0 * x0_64 + coef_x_1 * x1_64
    std64 = var.sqrt()

    return mean64.to(dtype), std64.to(dtype)


def bridge_kernel(
    x_t: Tensor, x_0: Tensor, t, t_prev, dm: NetCFM, eta: float = 1.0
) -> Tuple[Tensor, Tensor]:
    """
    Diffusion Model defined as::

        X_t = alpha_t * X_0 + sigma_t * Z

    For ``(alpha_t)_t`` and ``(sigma_t)_t`` to be a valid scheduler
        - they must be positive
        - ``(alpha_t)_t`` must be a deceasing function
        - ``(sigma_t)_t`` must be an increasing function
        - ``alpha_0 == 1`` and ``sigma_0 == 0``

    Returns
    -------
    mean, std : (Tensor, Tensor)
        mean and std of the bridge
    """
    x_dtype = x_t.dtype

    # TODO to be done formally
    # usage of float 64 is to increase precision as the coef are too close to zero
    alpha_t, alpha_t_prev = dm.alphas_f64[t], dm.alphas_f64[t_prev]
    sigma_t, sigma_t_prev = dm.sigmas_f64[t], dm.sigmas_f64[t_prev]

    a_t_t_prev = alpha_t / alpha_t_prev
    sig_t_prev_t = sigma_t_prev / sigma_t

    # NOTE the variance of the DDPM bridge
    var = sigma_t_prev**2 * (1 - (a_t_t_prev * sig_t_prev_t) ** 2)
    var = eta**2 * var

    coef_x_t = (sig_t_prev_t**2 - var / sigma_t**2).sqrt()
    coef_x_0 = alpha_t_prev - alpha_t * coef_x_t

    # cast into the input dtype dtype
    coef_x_t = coef_x_t.to(x_dtype)
    coef_x_0 = coef_x_0.to(x_dtype)
    std = var.sqrt().to(x_dtype)

    return coef_x_t * x_t + coef_x_0 * x_0, std


def ddim_sampler(initial_noise: Tensor, dm: NetCFM, eta: float = 1.0) -> Tensor:
    """DDIM sampler with ``eta`` to change how it is stochastic.

    Notes
    -----
    - When ``eta=0``: determinist sampling (also equivalent to simulating the PF-ODE)
    - When ``eta=1``: stochastic sampling as defined in DDPM (reverse of The OU SDE)
    """
    x_t = initial_noise
    for i in tqdm(range(len(dm.timesteps) - 1, 0, -1)):
        t, t_prev = dm.timesteps[i], dm.timesteps[i - 1]

        pred_x0 = dm.pred_x0(x_t, t)
        mean, std = bridge_kernel(x_t, pred_x0, t, t_prev, dm, eta)

        if eta != 0:
            x_t = mean + std * torch.randn_like(mean)
        else:
            x_t = mean

    return dm.pred_x0(x_t, t_prev)
