from tqdm import tqdm

import torch
from torch import Tensor

from fm.wrappers import NetCFM
from fm.inv_problem import InverseProblem

from image_utils import display


def pnp_flow(
    dm: NetCFM,
    inv_problem: InverseProblem,
    initial_noise: Tensor,
    n_steps: int = 50,
    lr: float = 1e-1,
    lr_style: str = "constant",
    alpha: float = 1.0,
    n_samples: int = 5,
    **kwargs,
):
    """PnP-Flow algorithm as described in [1].

    Implementation of Algorithm 3 based on the released code
    https://github.com/annegnx/PnP-Flow

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

    lr_style : str
        learning scheduler, see ``_get_learning_rate`` for possible values

    alpha : float
        power of the learning rate

    n_samples : int
        The number of iid samples for the averaging in the denoising step.

    Returns
    -------
    samples : Tensor
        Reconstructions

    References
    ----------
    .. [1] Martin, Ségolène, et al. "Pnp-flow: Plug-and-play image restoration with flow matching."
        ICLR, 2025.
    """

    log_pot, obs_std = inv_problem.log_pot, inv_problem.obs_std
    n_samples = initial_noise.shape[0]

    def _likelihood_wih_decoder(z):
        return log_pot(dm.decode(z))

    # This is used for initialization
    obs_img = inv_problem.obs_img

    lr_fn = _get_learning_rate(lr_style, alpha)
    base_lr = lr * obs_std**2

    # setup of the model
    dm.set_timesteps(n_steps)
    timesteps = dm.timesteps

    # ensure to account for the number of samples to generate
    encoded_im = dm.encode(obs_img)
    latent_x = encoded_im.repeat(n_samples, *[1 for _ in range(encoded_im.ndim - 1)])

    pbar = tqdm(enumerate(timesteps.flip(0)), desc="main loop")
    for step_idx, t_idx in pbar:

        # FM time: quantity between [0, 1]
        t = dm.sigmas[t_idx]

        # 1. forward step
        latent_x.requires_grad_()
        df_loss = -_likelihood_wih_decoder(latent_x)
        df_loss.backward()

        current_lr = lr_fn(base_lr, t)
        grad_df_loss = latent_x.grad
        with torch.no_grad():
            latent_x = latent_x - current_lr * grad_df_loss

            # NOTE the interpolation and backward steps are performed ``n_samples`` times
            # https://github.com/annegnx/PnP-Flow/blob/dbce1d96f7a6914bc2c581c73b2483848ec18ef7/pnpflow/methods/pnp_flow.py#L113-L119
            # 2. interpolation
            new_latent = torch.zeros_like(latent_x)
            for _ in range(n_samples):
                z_t = (1 - t) * latent_x + t * torch.randn_like(latent_x)

                # 3. backward step
                new_latent += dm.pred_x0(z_t, t_idx)
            latent_x = new_latent / n_samples

    # Return final reconstruction
    with torch.no_grad():
        return dm.decode(latent_x)


def _get_learning_rate(lr_style: str, alpha: float = 1.0):

    # NOTE in PNP-FLOW, t starts from 0 (Gaussian) ---> 1 (Data distribution)
    # whereas we adopt the convention t=0 (Data distribution) ---> t=1 (Gaussian)
    # hence the scheduler is inverted compared to reference implementation
    # https://github.com/annegnx/PnP-Flow/blob/dbce1d96f7a6914bc2c581c73b2483848ec18ef7/pnpflow/methods/pnp_flow.py#L28-L36
    gamma_styles = {
        "1_minus_t": lambda lr, t: lr * t,
        "sqrt_1_minus_t": lambda lr, t: lr * torch.sqrt(t),
        "constant": lambda lr, t: lr,
        "alpha_1_minus_t": lambda lr, t: lr * t**alpha,
    }

    return gamma_styles.get(lr_style, "constant")
