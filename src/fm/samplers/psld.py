from tqdm import tqdm

import torch
from torch import Tensor

from fm.wrappers import NetCFM
from fm.inv_problem import InverseProblem
from fm.bridge import bridge_kernel

from image_utils import display


def psld_sampler(
    dm: NetCFM,
    inv_problem: InverseProblem,
    initial_noise: Tensor,
    n_steps: int = 50,
    gamma: float = 0.1,
    omega: float = 0.1,
    eta: float = 1.0,
):
    """PSLD algorithm as described in [1].

    This is an implement of the Algorithm 2 in [1].
    Official implementation is available in https://github.com/LituRout/PSLD.
    In particular, https://github.com/LituRout/PSLD/blob/d734647bbc1ed0b1171521a804fc744973779f8c/stable-diffusion/ldm/models/diffusion/psld.py#L188-L338

    Parameters
    ----------
    dm: instance of SD3Wrapper
        Flow / Diffusion model

    inv_problem : Instance of InverseProblem
        Object that defines the inverse problem.

    initial_noise : Tensor
        initial noise

    gamma : float
        denoted by eta in the algo, stepsize associated with the constraint

    omega : float
        gamma in the algorithm, stepsize associated with likelihood.

    eta : float, default=1
        DDIM hyperparameter. If ``eta=1``, the sampling algorithm is DDPM.

    Returns
    -------
    samples : Tensor
        Reconstructions.

    References
    ----------
    .. [1] Rout, Litu, et al. "Solving linear inverse problems provably via
        posterior sampling with latent diffusion models."
        Advances in Neural Information Processing Systems 36 (2024).
    """
    n_samples = initial_noise.shape[0]
    obs, H_fn = inv_problem.obs, inv_problem.H_func

    if inv_problem.H_func.is_latent:
        im_shape = initial_noise.shape[1:]
    else:
        im_shape = (3, dm.im_size, dm.im_size)

    def _Ht_as_im(x):
        Ht_dot_x = H_fn.Ht(x)
        return Ht_dot_x.view(n_samples, *im_shape)

    # setup
    dm.set_timesteps(n_steps)
    timesteps = dm.timesteps.flip(0)

    Ht_obs = _Ht_as_im(obs)

    start_idx = 1
    z_t = initial_noise.clone()

    pbar = tqdm(enumerate(timesteps[start_idx:-1], start=start_idx), desc="main loop")
    for step_idx, t_idx in pbar:
        t_idx_prev = timesteps[step_idx + 1]

        z_t = z_t.requires_grad_()
        z_0t = dm.pred_x0(z_t, t_idx)
        decoded_z_0t = dm.decode(z_0t)
        H_dot_decoded_z_0t = H_fn.H(decoded_z_0t)

        # compute errors
        # NOTE: the official implementation of the algorithm uses norm (without square)
        #   yet the algorithm features the norm square
        #   doing so is equivalent to performing normalized gradient
        ll_error = torch.norm(obs - H_dot_decoded_z_0t)
        gluing_error = torch.norm(
            z_0t - dm.encode(Ht_obs + decoded_z_0t - _Ht_as_im(H_dot_decoded_z_0t))
        )

        error = omega * ll_error + gamma * gluing_error
        error.backward()
        grad = z_t.grad

        with torch.no_grad():
            mean, std = bridge_kernel(z_t, z_0t, t_idx, t_idx_prev, dm, eta)
            z_t = mean + std * torch.randn_like(mean)
            z_t = z_t - grad

    # Return final reconstruction
    with torch.no_grad():
        z_0t = dm.pred_x0(z_t, t_idx_prev)
        return dm.decode(z_0t)
