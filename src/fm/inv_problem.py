from typing import Callable

from dataclasses import dataclass
from pathlib import Path
from PIL import Image

import torch
import torchvision.transforms.v2 as tv_transforms
import numpy as np
from torch import Tensor
import torch.nn.functional as F

from fm.operators import OPERATORS


@dataclass
class InverseProblem:
    obs: Tensor
    obs_img: Tensor
    obs_std: float
    H_func: object
    log_pot: Callable[[Tensor], Tensor]
    x_ref: Tensor


def load_img(img, target_size, device, dtype, **kwargs):
    img_preprocessor = tv_transforms.Compose(
        [
            tv_transforms.Resize(target_size),
            tv_transforms.CenterCrop(target_size),
            tv_transforms.ToTensor(),
        ]
    )

    # XXX give the ability to provide an image as input to build inverse problems
    if not isinstance(img, str) and not isinstance(img, Path):
        # NOTE ``img`` is expected to be between [-1, 1]
        x_ref = img_preprocessor(img)
        x_ref.to(device, dtype)

    else:
        image = Image.open(img)

        x_ref = img_preprocessor(image)
        # map image to [-1, 1] interval
        x_ref = 2 * x_ref - 1
        x_ref = x_ref.to(device, dtype)

    return x_ref


def create_inv_prob(img, task: str, obs_std, device, dtype, **kwargs):
    """

    Parameters
    ----------
    img : str or Path or Tensor

    task : str
        Can be
        - blur
        - motion_blur
        - box_inpainting
        - high_dynamic_range
    """

    # load operator
    img_shape = img.shape[-1]
    H_func = OPERATORS.get(task, None)(img_shape, device=device, dtype=dtype, **kwargs)

    # --- generate observation and log_pot function
    obs = H_func.H(img.unsqueeze(0)).to(device)
    obs = obs + obs_std * torch.randn_like(obs)

    def log_pot(x: Tensor):
        diff = obs - H_func.H(x)
        return -0.5 * torch.norm(diff) ** 2 / obs_std**2

    # for plotting
    # some algos use it as initialization
    obs_img = H_func.get_obs_as_img(obs)

    return InverseProblem(
        obs,
        obs_img,
        obs_std,
        H_func,
        log_pot,
        img,
    )
