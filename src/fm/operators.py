import torch
import numpy as np
from torch import Tensor

OPERATORS = {}


def register_operator(name: str):
    def wrapper(cls):
        if OPERATORS.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        OPERATORS[name] = cls
        return cls

    return wrapper


@register_operator("box_inpainting")
class BoxInpainting:
    """Fast box inpainting operator that inherits directly from H_functions."""

    def __init__(
        self,
        im_shape=768,
        box=(84, 422, 253, 422),
        is_latent=False,
        latent_factor: int = 8,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__()
        assert all(box_el <= im_shape for box_el in box)

        self.im_shape = im_shape
        self.device = device
        self.dtype = dtype
        self.is_latent = is_latent
        self.latent_factor = latent_factor

        # rectangular mask
        x_start, x_end = box[:2]
        y_start, y_end = box[2:]

        self.mask = torch.ones(im_shape, im_shape, dtype=torch.bool, device=device)
        self.mask[x_start:x_end, y_start:y_end] = False

        self.mask_3d = self.mask.view(1, 1, im_shape, im_shape)
        if is_latent:
            self.pixel_mask = torch.nn.functional.interpolate(
                self.mask_3d.to(dtype),
                im_shape * latent_factor,
                mode="nearest",
            )
        else:
            self.pixel_mask = self.mask_3d.clone().to(dtype)

    def H(self, x: Tensor):
        result = x * self.mask_3d
        return result  # .reshape(x.shape[0], -1)

    def Ht(self, x: Tensor):
        """Transpose of inpainting operator: same as H for this case."""
        return self.H(x)

    def get_obs_as_img(self, obs):
        obs_img = obs.reshape((1, -1, self.im_shape, self.im_shape))
        obs_img = obs_img  # .clamp(-1.0, 1.0)
        return obs_img


@register_operator("mask_inpainting")
class MaskInpainting(BoxInpainting):
    def __init__(
        self,
        im_shape=768,
        path_mask: str = None,
        latent_factor: int = 8,
        is_latent: bool = True,
        device="cpu",
        dtype=torch.float32,
    ):
        self.path_mask = path_mask
        self.latent_factor = latent_factor

        # box given as placeholder to the constructor is a placeholder
        is_latent = True
        super().__init__(
            im_shape,
            box=(0, 1, 0, 1),
            latent_factor=latent_factor,
            is_latent=is_latent,
            device=device,
            dtype=dtype,
        )

        # handle case numpy file
        path_mask = str(path_mask)
        if path_mask.endswith(".npy"):
            mask = torch.from_numpy(np.load(path_mask)).to(dtype, device)
        else:
            mask = torch.load(path_mask, map_location=device).to(dtype)

        self.mask_3d = self._get_mask_latent(mask)
        self.pixel_mask = mask

    def _get_mask_latent(self, mask_pixel):
        mask_size = mask_pixel.shape[-1]

        # ensure to have the right shape (batch and channels)
        mask_pixel = mask_pixel.reshape(1, 1, mask_size, mask_size)

        mask_resized = torch.nn.functional.interpolate(
            mask_pixel,
            mask_size // self.latent_factor,
            mode="bilinear",
            antialias=True,
        )
        # over-estimate the mask to avoid artifacts
        mask_resized = (mask_resized > 0.95).to(self.dtype)

        return mask_resized
