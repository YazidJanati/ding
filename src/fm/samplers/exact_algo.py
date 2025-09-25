from tqdm import tqdm

import torch
from torch import Tensor

from fm.wrappers import NetCFM
from fm.operators import BoxInpainting
from fm.bridge import bridge_kernel
from fm.inv_problem import InverseProblem


def ding(
    dm: NetCFM,
    inv_problem: InverseProblem,
    initial_noise: Tensor,
    n_steps: int = 50,
    eta_type: str = "default",
):
    H_func = inv_problem.H_func
    if not isinstance(H_func, BoxInpainting) and not H_func.is_latent:
        raise ValueError("Operator must be latent Inpainting")

    x_dtype = initial_noise.dtype
    mask = H_func.mask_3d
    obs, obs_std = inv_problem.obs, inv_problem.obs_std

    dm.set_timesteps(n_steps)
    timesteps = dm.timesteps.flip(0)

    # NOTE: skip the last three iterations as sigma becomes zero
    # because of finite precision (bfloat16)
    x_t = torch.randn_like(initial_noise)
    pbar = tqdm(enumerate(timesteps[:-2]), desc="Main Loop")
    for step_idx, t_idx in pbar:

        t_prev_idx = timesteps[step_idx + 1]
        alpha_t_prev, sigma_t_prev = dm.alphas[t_prev_idx], dm.sigmas[t_prev_idx]
        eta = _get_eta(t_idx, t_prev_idx, dm, x_dtype, eta_type=eta_type)

        pred_x_0 = dm.pred_x0(x_t, t_idx)
        mean_prior, std_prior = bridge_kernel(
            x_t, pred_x_0, t_idx, t_prev_idx, dm, eta=eta
        )
        z_s = mean_prior + std_prior * torch.randn_like(mean_prior)

        noise_pred = z_s + alpha_t_prev * dm.pred_velocity(z_s, t_prev_idx)
        x_t = _sample_posterior(
            obs=alpha_t_prev * obs + sigma_t_prev * (mask * noise_pred),
            mask=mask,
            obs_std=obs_std * alpha_t_prev,
            mean_prior=mean_prior,
            std_prior=std_prior,
        )

    x_0 = dm.pred_x0(x_t, t_prev_idx)
    return dm.decode(x_0)


def _sample_posterior(
    obs: Tensor, mask: Tensor, obs_std: float, mean_prior: Tensor, std_prior: float
):
    # sample from the posterior ``p(y|x) p(x)`` where
    #  - p(y|x) = N(y; diag(mask) x, diag(obs_std**2))
    #  - p(x)   = N(x; mean_prior, diag(std_prior**2))

    # NOTE: `mask` consists of 0s and 1s, hence no need to square it
    #   do so if it is not the case
    cov = 1 / ((mask / obs_std**2) + (1 / std_prior**2))
    unscaled_mean = (mask * obs) / obs_std**2 + mean_prior / std_prior**2

    return cov * unscaled_mean + cov.sqrt() * torch.randn_like(mean_prior)


def _get_eta(t_idx, t_prev_idx, dm: NetCFM, x_dtype, eta_type):

    alpha_t, alpha_t_prev = dm.alphas_f64[t_idx], dm.alphas_f64[t_prev_idx]
    sigma_t, sigma_t_prev = dm.sigmas_f64[t_idx], dm.sigmas_f64[t_prev_idx]

    a_t_t_prev = alpha_t / alpha_t_prev
    sig_t_prev_t = sigma_t_prev / sigma_t

    if eta_type == "default":
        # --- magic eta for ddim bridge
        eta = ((1 - alpha_t_prev) / (1 - (a_t_t_prev * sig_t_prev_t) ** 2)).sqrt()

    elif eta_type == "square":
        eta = (
            (1 - alpha_t_prev).square() / (1 - (a_t_t_prev * sig_t_prev_t) ** 2)
        ).sqrt()

    elif eta_type == "max":
        eta = min(
            1 / ((1 - (a_t_t_prev * sig_t_prev_t) ** 2)).sqrt(), torch.tensor(1.0)
        )

    elif eta_type == "ddpm":
        eta = torch.tensor(1.0)

    elif eta_type == "ddim":
        eta = torch.tensor(1e-2)

    # eta = eta.to(x_dtype)
    return eta
