from tqdm import tqdm
import numpy as np

import torch
from torch import Tensor

from fm.wrappers import NetCFM
from fm.inv_problem import InverseProblem
from fm.bridge import bridge_kernel


def daps_sampler(
    dm: NetCFM,
    inv_problem: InverseProblem,
    initial_noise: Tensor,
    n_steps: int = 50,
    n_diffusion_steps: int = 2,
    # ---
    tau: float = 1e-2,
    # --- mcmc hyperparams
    mcmc_sampler: str = "Langevin",
    n_mcmc_steps: int = 20,
    lr: float = 1.11e-6,
    min_ratio: float = 0.43,
    mcmc_sampler_extra_params={},
    # ---
):
    """DAPS algorithm as described in [1].

    Implementation of Algorithm for FM models.
    Code adapted from official repository
    https://github.com/zhangbingliang2019/DAPS

    Parameters
    ----------
    dm: instance of SD3Wrapper
        Flow / Diffusion model.

    inv_problem : Instance of InverseProblem
        Object that defines the inverse problem.

    initial_noise : Tensor
        initial noise

    n_steps : int
        The number of annealing steps. Equivalent to ``n_annealing_steps`` in our DAPS
        official implementation.

    n_diffusion_steps : int
        The number of diffusion steps used to estimate ``x_0t``.

    tau : float
        a scaling factor of the likelihood. Note that DAPS doesn't account for the observation std in the likelihood. The likelihood will be divided by ``tau**2``.

    mcmc_sampler : str
        The name of the MCMC sampler either 'Langevin' or 'HMC'.

    lr : float
        stepsize of the MCMC sampler.

    min_ratio : float
        the stepsize will be multiplied along the annealing steps by a factor that
        interpolates ``[min_ratio, 1]``.

    mcmc_sampler_extra_params : Dict
        Extra parameters to be pass to the MCMC sampler.

    Returns
    -------
    samples : Tensor
        Reconstructions.

    References
    ----------
    .. [1] Zhang, Bingliang, Wenda Chu, Julius Berner, Chenlin Meng, Anima Anandkumar, and Yang Song.
        "Improving diffusion inverse problem solving with decoupled noise annealing."
        arXiv preprint arXiv:2407.01521 (2024).
    """

    obs, H_fn = inv_problem.obs, inv_problem.H_func

    # setup
    dm.set_timesteps(n_steps)
    timesteps = dm.timesteps.flip(0)

    mcmc_sampler_fn = globals().get(mcmc_sampler)(
        initial_noise.shape, **mcmc_sampler_extra_params
    )

    start_idx = 1
    x_t = initial_noise.clone()

    pbar = tqdm(enumerate(timesteps[start_idx:-1], start=start_idx), desc="main loop")
    for step_idx, t_idx in pbar:
        t_idx_prev = timesteps[step_idx + 1]

        # 1. estimate x_0
        current_timesteps = torch.linspace(
            0, t_idx, n_diffusion_steps + 1, dtype=torch.int32
        )
        x_0t = _sample_x0t(x_t, dm, current_timesteps, eta=0.0)

        # 2. MCMC to conditional on the obs
        x_0_i = x_0t.clone()
        # XXX heuristic in DAPS, in VE, sigma is used instead
        std_prior = dm.sigmas[t_idx] / dm.alphas[t_idx]
        current_lr = _get_lr(step_idx / n_steps, lr, min_ratio)
        for _ in range(n_mcmc_steps):
            score_prior = -(x_0_i - x_0t) / std_prior**2

            x_0_i.requires_grad_()
            llh = -0.5 * ((obs - H_fn.H(dm.decode(x_0_i))) / tau).norm() ** 2
            llh.backward()
            score_llh = x_0_i.grad

            with torch.no_grad():
                x_0_i = mcmc_sampler_fn(x_0_i, score_prior + score_llh, current_lr)

        # 3. re-noise
        noise = torch.randn_like(x_t)
        x_t = dm.alphas[t_idx_prev] * x_0_i + dm.sigmas[t_idx_prev] * noise

    # Return final reconstruction
    with torch.no_grad():
        return dm.decode(x_t)


# copy/paste of https://github.com/zhangbingliang2019/DAPS/blob/1e319581754ec9689f20d717d56313f65f1a7409/cores/mcmc.py#L174-L181
def _get_lr(ratio, lr, lr_min_ratio):
    p = 1
    multiplier = (1 ** (1 / p) + ratio * (lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
    return multiplier * lr


def _sample_x0t(x_t, dm: NetCFM, timesteps, eta: float = 1.0) -> Tensor:

    for i in range(len(timesteps) - 1, 0, -1):
        t, t_prev = timesteps[i], timesteps[i - 1]

        pred_x0 = dm.pred_x0(x_t, t)
        mean, std = bridge_kernel(x_t, pred_x0, t, t_prev, dm, eta)

        if eta != 0:
            x_t = mean + std * torch.randn_like(mean)
        else:
            x_t = mean

    return dm.pred_x0(x_t, t_prev)


class Langevin:
    def __init__(self, x_shape):
        self.x_shape = x_shape

    def __call__(self, x, score, lr):
        noise = torch.randn(*self.x_shape)
        return x + lr * score + np.sqrt(2 * lr) * noise


class HMC:
    def __init__(self, x_shape, momentum):
        self.x_shape = x_shape
        self.momentum = momentum
        self.velocity = torch.randn(x_shape)

    def __call__(self, x_t, score, lr):
        noise = torch.randn(*self.x_shape)

        step_size = np.sqrt(lr)
        self.velocity = (
            self.momentum * self.velocity
            + step_size * score
            + np.sqrt(2 * (1 - self.momentum)) * noise
        )

        return x_t + self.velocity * step_size
