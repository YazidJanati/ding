from tqdm import tqdm
import torch
from torch import Tensor
from fm.wrappers import NetCFM
from fm.inv_problem import InverseProblem
from typing import List, Optional
from image_utils import display


def flow_dps(
    dm: NetCFM,
    inv_problem: InverseProblem,
    initial_noise: Tensor,
    n_steps: int,
    step_size: float = 8.0,
    dc_iters: int = 3,
    latent: Optional[torch.Tensor] = None,
    show_every: Optional[int] = None,
    show_intermediate: bool = True,
):
    """flowDPS algorithm as described in [1].

    Implementation of Algorithm 1 based on the released code
    https://github.com/FlowDPS-Inverse/FlowDPS

    Parameters
    ----------
    dm : instance of SD3Wrapper
        Diffusion/Flow model

    inv_problem : Instance of InverseProblem
        Object that defines the inverse problem.

    n_steps : int
        The number of steps to perform

    step_size : float
        Gradient step size for data consistency in latent space

    dc_iters : int
        Number of inner data-consistency optimization iterations per outer step

    latent : Optional[torch.Tensor]
        Initial latent. If None, sampled from N(0, I).

    show_every : int
        frequency of showing intermediate results

    Returns
    -------
    samples : Tensor
        Reconstructions

    References
    ----------
    .. [1] Kim, Jeongsol, et al. "FlowDPS : Flow-driven posterior sampling for inverse problems."
        ICCV, 2025.
    """

    obs, H_fn = inv_problem.obs, inv_problem.H_func
    dm.set_timesteps(n_steps)
    timesteps = dm.timesteps.flip(0)

    # ---- init latent ----
    z = (
        torch.randn_like(initial_noise)
        if latent is None
        else latent.to(dm.device, dm.dtype)
    )

    if show_every is None:
        show_every = max(1, n_steps // 5)

    # ---- main loop ----
    for i in tqdm(range(len(timesteps) - 1), desc="SD3-FlowDPS"):

        t_idx = timesteps[i]
        t_prev = timesteps[i + 1]
        sigma = dm.sigmas[t_idx].to(dm.dtype)

        sigma_prev = dm.sigmas[t_prev].to(dm.dtype)

        alpha = 1.0 - sigma

        # v-pred (+CFG) at index i
        with torch.no_grad():
            v_cfg = dm.pred_velocity(z, t_idx)
            z0t = z - sigma * v_cfg  # clean latent
            z1t = z + alpha * v_cfg  # noise latent

        # ---- data consistency on clean latent  ----
        z_dc = z0t
        for _ in range(dc_iters):
            z_dc = z_dc.detach().requires_grad_(True)
            x0t = dm.decode(z_dc)
            loss = torch.linalg.norm((H_fn.Ht(obs) - H_fn.Ht(H_fn.H(x0t))).view(1, -1))

            ### In the main repo, they didn’t define the loss as a squared norm, so they use a high step size  but apply pseudo inverse

            grad = torch.autograd.grad(loss, z_dc)[0]
            z_dc = (z_dc - step_size * grad.to(z_dc.dtype)).detach()

        # Blend DC result with clean per FlowDPS rule
        z0_blend = alpha * z0t + sigma * z_dc

        # Re-noise & update latent
        eps = torch.randn_like(z1t)
        noise = torch.sqrt(sigma_prev) * z1t + torch.sqrt(1.0 - sigma_prev) * eps
        z = (1.0 - sigma_prev) * z0_blend + sigma_prev * noise

        # ---- optional visualization ----
        if show_intermediate and (
            ((n_steps - i) % show_every) == 0 or i == n_steps - 1
        ):
            decode_fn = (
                dm.decode if not inv_problem.H_func.is_latent else dm.pixel_decode
            )
            x_img = decode_fn(z)

            display(
                x_img.clamp(-1, 1).cpu(),
                title=f"FlowDPS | step {n_steps - i}",
            )

    # ---- final decode ----
    with torch.no_grad():
        img = dm.decode(z)
    return img
