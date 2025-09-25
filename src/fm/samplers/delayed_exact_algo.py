from tqdm import tqdm

import torch
from torch import Tensor

from fm.wrappers import NetCFM
from fm.operators import BoxInpainting
from fm.bridge import bridge_kernel
from fm.inv_problem import InverseProblem


def delayed_ding(
    dm: NetCFM,
    inv_problem: InverseProblem,
    initial_noise: Tensor,
    n_steps: int = 50,
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

        alpha_t, sigma_t = dm.alphas[t_idx], dm.sigmas[t_idx]
        alpha_t_prev, sigma_t_prev = dm.alphas[t_prev_idx], dm.sigmas[t_prev_idx]
        eta = _get_eta(t_idx, t_prev_idx, dm, x_dtype)

        v_pred = dm.pred_velocity(x_t, t_idx)
        pred_x_0, pred_x_1 = (
            x_t - sigma_t * v_pred,
            x_t + alpha_t * v_pred,
        )
        mean_prior, std_prior = bridge_kernel(
            x_t, pred_x_0, t_idx, t_prev_idx, dm, eta=eta
        )
        x_t = _sample_posterior(
            obs=alpha_t_prev * obs + sigma_t_prev * (mask * pred_x_1),
            mask=mask,
            obs_std=obs_std * alpha_t_prev,
            mean_prior=mean_prior,
            std_prior=std_prior,
        )

        # # XXX uncomment for debugging
        # # Plot every xx steps
        # decode_fn = dm.decode if not inv_problem.H_func.is_latent else dm.pixel_decode
        # if step_idx % 10 == 0:
        #     z_0t = dm.pred_x0(x_t[[0]], t_prev_idx)
        #     im = decode_fn(z_0t)
        #     display(im)

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


def _get_eta(t_idx, t_prev_idx, dm: NetCFM, x_dtype):
    # --- magic eta for ddim bridge
    alpha_t, alpha_t_prev = dm.alphas_f64[t_idx], dm.alphas_f64[t_prev_idx]
    sigma_t, sigma_t_prev = dm.sigmas_f64[t_idx], dm.sigmas_f64[t_prev_idx]

    a_t_t_prev = alpha_t / alpha_t_prev
    sig_t_prev_t = sigma_t_prev / sigma_t

    # eta = ((1 - alpha_t**2) / (1 - (a_t_t_prev * sig_t_prev_t) ** 2)).sqrt()
    eta = ((1 - alpha_t_prev).square() / (1 - (a_t_t_prev * sig_t_prev_t) ** 2)).sqrt()
    eta = eta.to(x_dtype)
    return eta
