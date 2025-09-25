from tqdm import tqdm
from typing import Callable

import torch
from torch import Tensor

from fm.wrappers import NetCFM
from fm.inv_problem import InverseProblem
from fm.bridge import bridge_kernel

from image_utils import display


def resample_sampler(
    dm: NetCFM,
    inv_problem: InverseProblem,
    initial_noise: Tensor,
    n_steps: int,
    use_dps_cond: bool = True,
    frequency: int = 10,
    max_iters: int = 200,
    sigma_scale: float = 40.0,
    lr_pixel: float = 1e-2,
    lr_latent: float = 5e-3,
):
    """Resample algorithm as introduced in [1].

    Implementation combines the Algo 1 in [1] and details
    in appendix C1 with the official implementation in
    https://github.com/soominkwon/resample

    Parameters
    ----------
    dm : instance of SD3Wrapper
        Diffusion/Flow model

    inv_problem : Instance of InverseProblem
        Object that defines the inverse problem.

    n_steps : int
        The number of steps to perform.

    use_dps_cond : bool
        Whether to add DPS conditioning in DDIM steps.

    frequency : int
        The hard data consistency will be applied every ``frequency`` step.

    max_iters : int
        Max number of iteration used in the optimization.

    sigma_scale : float
        The scale of the variance in the stochastic resampling step.

    lr_pixel : float
        The learning rate of the pixel optimization.

    lr_latent : float
        The learning rate of the latent optimization.

    References
    ----------
    .. [1] Song, Bowen, et al. "Solving inverse problems with latent diffusion models via hard data consistency."
    arXiv preprint arXiv:2307.08123 (2023).
    """
    # NOTE: the following were hard-coded in the official implementation
    #   - lr are; 1e-2 / 5e-3 for pixel / latent
    #   - DPS scale is set to ``0.5 * 0.3`` without normalization by the loss
    #   - DDIM is being applied with eta=0 (deterministic)

    obs, H_fn = inv_problem.obs, inv_problem.H_func
    eps = inv_problem.obs_std

    dm.set_timesteps(n_steps)
    timesteps = dm.timesteps.flip(0)

    # losses used in hard data consistency
    pixel_loss_fn = lambda var: (obs - H_fn.H(var)).norm() ** 2
    latent_loss_fn = lambda var: (obs - H_fn.H(dm.decode(var))).norm() ** 2

    # NOTE defines three stages, see Appendix C1
    #   - stage 1: unconditional sampling with optionally DPS guidance
    #   - stage 2: stage 1 + pixel space consistency with frequency
    #   - stage 3: stage 1 + latent space consistency with frequency
    stage_size = n_steps // 3

    z_t = initial_noise
    pbar = tqdm(enumerate(timesteps[:-1]), desc="Main Loop")
    for step_idx, t_idx in pbar:

        t_prev_idx = timesteps[step_idx + 1]
        alpha_t, sigma_t = dm.alphas[t_idx], dm.sigmas[t_idx]
        alpha_t_prev, sigma_t_prev = dm.alphas[t_prev_idx], dm.sigmas[t_prev_idx]

        if use_dps_cond:
            z_t.requires_grad_()
            z_0 = dm.pred_x0(z_t, t_idx)

            loss = -0.5 * (obs - H_fn.H(dm.decode(z_0))).norm() ** 2
            loss.backward()

            scaled_grad = 0.5 * 0.3 * alpha_t**2 * z_t.grad
            with torch.no_grad():
                z_t, _ = bridge_kernel(z_t, z_0, t_idx, t_prev_idx, dm, eta=0.0)
                z_t += scaled_grad
            z_0.detach_()
        else:
            z_0 = dm.pred_x0(z_t, t_idx)
            z_t, _ = bridge_kernel(z_t, z_0, t_idx, t_prev_idx, dm, eta=0.0)

        # stage 2-3: perform measurement consistency
        if step_idx % frequency == 1:

            # pixel space
            if stage_size <= step_idx < 2 * stage_size:
                x = _optimization_loss(
                    init=dm.decode(z_0),
                    loss_fn=pixel_loss_fn,
                    stop_criterion=eps**2,
                    max_iters=max_iters,
                    lr=lr_pixel,
                )
                z_0 = dm.encode(x)
            # latent space
            elif step_idx >= 2 * stage_size:
                z_0 = _optimization_loss(
                    init=z_0,
                    loss_fn=latent_loss_fn,
                    stop_criterion=eps**2,
                    max_iters=max_iters,
                    lr=lr_latent,
                )
            else:
                continue

            # stochastic resampling
            # here gamma_t is equivalent to sigma_t**2 in Resample, Appendix C1
            # it is defined by scaling the DDPM transition variance
            a_t_t_prev = alpha_t / alpha_t_prev
            sig_t_prev_t = sigma_t_prev / sigma_t
            ddpm_variance = sigma_t_prev**2 * (1 - (a_t_t_prev * sig_t_prev_t) ** 2)

            gamma_t = sigma_scale * ddpm_variance
            z_t = _stochastic_resampling(z_t, z_0, alpha_t, sigma_t, gamma_t)

        # # XXX uncomment for debugging
        # # Plot every xx steps
        # decode_fn = dm.decode if not inv_problem.H_func.is_latent else dm.pixel_decode
        # if step_idx % 30 == 0:
        #     z_0t = dm.pred_x0(z_t[[0]], t_prev_idx)
        #     im = decode_fn(z_0t)
        #     display(im)

    # # NOTE the official implementation of Resample perform
    # # a latent optimization in the last step
    # loss_fn = lambda var: (obs - H_fn.H(dm.decode(var))).norm() ** 2
    # z_0 = _optimization_loss(
    #     init=z_0,
    #     loss_fn=loss_fn,
    #     stop_criterion=eps,
    #     max_iters=max_iters,
    #     lr=lr_latent,
    # )

    z_0 = dm.pred_x0(z_t, t_prev_idx)
    return dm.decode(z_0)


def _optimization_loss(
    init: Tensor,
    loss_fn: Callable[[Tensor], Tensor],
    stop_criterion: float,
    max_iters: int,
    lr: float,
):
    var = init.clone()
    var.requires_grad_()

    optimizer = torch.optim.AdamW([var], lr)
    for _ in range(max_iters):
        optimizer.zero_grad()

        current_loss = loss_fn(var)
        current_loss.backward()

        optimizer.step()

        if current_loss.item() < stop_criterion:
            break

    var.detach_()
    return var


def _stochastic_resampling(z_t, z_0, alpha_t, sigma_t, gamma_t):
    denominator = gamma_t + sigma_t**2
    std = (sigma_t**2 * gamma_t / denominator).sqrt()

    coef_z_t = sigma_t**2 / denominator
    coef_z_0 = gamma_t * alpha_t / denominator

    return coef_z_t * z_t + coef_z_0 * z_0 + std * torch.randn_like(z_t)
