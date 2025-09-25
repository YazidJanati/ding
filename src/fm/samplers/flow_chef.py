from tqdm import tqdm
import torch
from torch import Tensor
from fm.wrappers import NetCFM
from fm.inv_problem import InverseProblem
from typing import Optional
from image_utils import display


def flow_chef(
    dm: NetCFM,
    inv_problem: InverseProblem,
    initial_noise: Tensor,
    n_steps: int,
    step_size: float = 0.5,
    dc_iters: int = 3,
    latent: Optional[torch.Tensor] = None,
    show_every: Optional[int] = None,
    show_intermediate: bool = True,
):
    """FlowChef algorithm as described in [1].

    Implementation of Algorithm 1 based on the released code of FlowDPS
    https://github.com/FlowDPS-Inverse/FlowDPS & https://github.com/FlowChef/flowchef

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
    .. [1] Patel, Maitreya, et al. "Steering rectified flow models in the vector field for controlled image generation."  ICCV, 2025.
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

    # ---- main loop  ----
    for i in tqdm(range(len(timesteps) - 1), desc="SD3-FlowChef"):

        t_idx = timesteps[i]
        t_prev = timesteps[i + 1]
        sigma = dm.sigmas[t_idx].to(dm.dtype)

        sigma_prev = dm.sigmas[t_prev].to(dm.dtype)

        alpha = 1.0 - sigma

        # v-pred (+CFG) at step i
        with torch.no_grad():
            v_cfg = dm.pred_velocity(z, t_idx)
            z0t = z - sigma * v_cfg  # clean latent
            z1t = z + alpha * v_cfg  # noise latent

        # ---- data-consistency gradient ----
        z_dc = z0t.detach().clone()
        for _ in range(dc_iters):
            z_dc = z_dc.requires_grad_(True)
            x0t = dm.decode(z_dc)

            loss = torch.linalg.norm((H_fn.Ht(obs) - H_fn.Ht(H_fn.H(x0t))).view(1, -1))

            ### In the main repo, they didn’t define the loss as a squared norm but apply pseudo inverse

            grad = torch.autograd.grad(
                loss, z_dc, retain_graph=False, create_graph=False
            )[0]
            z_dc = (z_dc - step_size * grad).detach()

        # ---- FlowChef update ----
        z = z_dc + sigma_prev * (z1t - z0t)

        # ---- optional visualization ----
        if show_intermediate and (
            ((n_steps - i) % show_every) == 0 or i == n_steps - 1
        ):
            decode_fn = (
                dm.decode if not inv_problem.H_func.is_latent else dm.pixel_decode
            )
            x_img = decode_fn(z)

            ### In the main repo, they didn’t define the loss as a squared norm but apply pseudo inverse

            display(
                x_img.clamp(-1, 1).cpu(),
                title=f"FlowChef | step {n_steps - i}",
            )

    # ---- final decode ----
    with torch.no_grad():
        img = dm.decode(z)
    return img
