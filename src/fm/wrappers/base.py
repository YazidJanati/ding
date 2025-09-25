import torch
from torch import Tensor


class NetCFM(torch.nn.Module):
    """Abstraction that severs as base for all wrappers."""

    def __init__(
        self,
        alphas: Tensor,
        sigmas: Tensor,
        num_train_timesteps: int,
        dtype,
        device: str,
        call_super: bool = True,
    ):
        # HACK ``call_super`` has two purposes
        #   - re-init the object without losing the nn weights (e.g. saved by children)
        #   - to perform the preprocessing of alphas/sigmas (conversion ...)
        if call_super:
            super().__init__()

        self.dtype = dtype
        self.device = device

        self.alphas_f64 = alphas.to(device, torch.float64)
        self.sigmas_f64 = sigmas.to(device, torch.float64)

        self.alphas = alphas.to(device, dtype)
        self.sigmas = sigmas.to(device, dtype)
        self.num_train_timesteps = num_train_timesteps
        self.timesteps: Tensor

    def pred_x0(self, x, t):
        v_pred = self.pred_velocity(x, t)
        return x - self.sigmas[t] * v_pred

    # implemented by the model
    def pred_velocity(self, x, t):
        raise

    def score(self, x, t):
        raise

    def pred_x1(self, x, t):
        v_pred = self.pred_velocity(x, t)
        return x + self.alphas[t] * v_pred

    def set_timesteps(self, n_steps):
        self.timesteps = torch.linspace(
            0, self.num_train_timesteps - 1, n_steps, dtype=torch.int32
        )

    # by default encode and decode are set to the identity
    def encode(self, x):
        return x

    def decode(self, x):
        return x

    # for compatibility with torch
    def forward(self, x, t):
        return self.pred_x0(x, t)
