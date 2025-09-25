from tqdm import tqdm

import torch
from torch import Tensor

from fm.wrappers import NetCFM
from fm.inv_problem import InverseProblem

from image_utils import display


def reddiff_sampler(
    dm: NetCFM,
    inv_problem: InverseProblem,
    initial_noise: Tensor,
    n_steps: int,
    lr: float = 3e-2,
    sigma_x0: float = 1e-3,  # jitter on x0_pred
    grad_term_weight: float = 0.25,
    obs_weight: float = 1.0,
    denoise_weight_mode: str = "linear",  # {'linear','sqrt','square','log','trunc_linear','power2over3','const'}
    show_every: int = 10,
):
    """reddiff algorithm as described in [1].

    Implementation of Algorithm 1 based on the released code
    https://github.com/NVlabs/RED-diff

    Parameters
    ----------
    dm : instance of SD3Wrapper
        Diffusion/Flow model

    inv_problem : Instance of InverseProblem
        Object that defines the inverse problem.

    n_steps : int
        The number of steps to perform

    lr : float
        learning rate

    sigma_x0 : float
        jitter on x0_pred (similar to the main repository code)

    grad_term_weight : float
        a constant weight for the regularization term in the loss

    obs_weight : float
        a constant weight for the data fidelity term in the loss

    denoise_weight_mode : str
        different strategies for weighting the denoising (regularization) term in the loss based on SNR (lambda in the paper)

    show_every : int
        frequency of showing intermediate results

    Returns
    -------
    samples : Tensor
        Reconstructions

    References
    ----------
    .. [1] Mardani, Morteza, et al. "A variational perspective on solving inverse problems with diffusion models."
        ICLR, 2024.
    """
    device = dm.device
    n_samples = initial_noise.shape[0]

    obs, H_fn = inv_problem.obs, inv_problem.H_func
    # NOTE used for initialization
    obs_img = inv_problem.obs_img

    dm.set_timesteps(n_steps)
    ts = dm.timesteps

    assert ts.ndim == 1 and len(ts) >= 3

    # ensure to account for the number of samples to generate
    encoded_im = dm.encode(obs_img)
    mu = encoded_im.repeat(n_samples, *[1 for _ in range(encoded_im.ndim - 1)])
    mu.requires_grad_()

    opt = torch.optim.Adam([mu], lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    pbar = tqdm(range(len(ts) - 1, 0, -1), desc="REDDIFF (noise-based)")
    for i in pbar:
        t_idx = ts[i]

        alpha_t = dm.alphas[t_idx]
        sigma_t = dm.sigmas[t_idx]

        noise_x0 = torch.randn_like(mu)
        noise_xt = torch.randn_like(mu)

        x0_pred = mu + sigma_x0 * noise_x0
        x_t = alpha_t * x0_pred + sigma_t * noise_xt

        with torch.no_grad():
            eps_pred = dm.pred_x1(x_t, int(t_idx.item()))

        eps_scalar = torch.tensor(
            torch.finfo(mu.dtype).eps, device=device, dtype=mu.dtype
        )
        snr_inv = (sigma_t / (alpha_t + eps_scalar)).squeeze()

        if denoise_weight_mode == "linear":
            w_scale = snr_inv
        elif denoise_weight_mode == "sqrt":
            w_scale = snr_inv.sqrt()
        elif denoise_weight_mode == "square":
            w_scale = snr_inv**2
        elif denoise_weight_mode == "log":
            w_scale = (snr_inv + 1.0).log()
        elif denoise_weight_mode == "trunc_linear":
            w_scale = snr_inv.clamp_max(1.0)
        elif denoise_weight_mode == "power2over3":
            w_scale = snr_inv ** (2.0 / 3.0)
        elif denoise_weight_mode == "const":
            w_scale = torch.ones_like(snr_inv)
        else:
            w_scale = snr_inv

        w_t = torch.tensor(
            float(grad_term_weight) * float(w_scale.item()),
            device=device,
            dtype=torch.float32,
        )
        v_t = torch.tensor(float(obs_weight), device=device, dtype=torch.float32)

        # ----- losses

        loss_noise = ((eps_pred - noise_xt).detach() * x0_pred).mean()

        # Data term in image space
        x_img = dm.decode(x0_pred)
        Hx = H_fn.H(x_img)
        e_obs = obs - Hx
        loss_obs = 0.5 * (e_obs**2).mean()

        # Accumulate in fp32 for stability
        loss = (w_t * loss_noise.float()) + (v_t * loss_obs.float())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        pbar.set_postfix({"L_obs": f"{loss_obs.item():.3e}"})

    mu.detach_()
    x_final = dm.decode(mu)
    return x_final
