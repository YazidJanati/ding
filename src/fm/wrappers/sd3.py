from typing import List
from pathlib import Path

import torch
from torch import Tensor
from diffusers import StableDiffusion3Pipeline, AutoencoderTiny
from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetModel

from fm.wrappers.base import NetCFM


class SD3Wrapper(NetCFM):

    def __init__(
        self,
        device: str,
        pipeline=None,
        dtype=torch.float16,
        controlnet=None,
    ):

        if isinstance(pipeline, (str, Path)):
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                pipeline, torch_dtype=dtype
            ).to(device)
        else:
            self.pipeline = pipeline

        # NOTE here the scheduler are
        #   alphas = 1 - t
        #   sigmas = t
        self.scheduler = self.pipeline.scheduler
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)
        self.net_timesteps = self.scheduler.timesteps.flip(0)
        self.net_timesteps = self.net_timesteps.to(device, dtype)

        sigmas_f64 = self.scheduler.sigmas.flip(0).to(torch.float64)
        alphas_f64 = 1 - sigmas_f64

        # init model
        super().__init__(
            alphas_f64,
            sigmas_f64,
            self.scheduler.config.num_train_timesteps,
            dtype,
            device,
        )

        # NOTE for controlnet sampler only: to avoid re-loading controlnet each time
        self.controlnet = None
        if isinstance(controlnet, (str, Path)):
            self.controlnet = SD3ControlNetModel.from_pretrained(
                controlnet,
                use_safetensors=True,
                extra_conditioning_channels=1,
            ).to(device, dtype)

        # NOTE this must go after as these are torch networks
        # and torch needs to track them
        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae

        # set networks to inference mode
        for net in (
            self.transformer,
            self.vae,
            self.pipeline.text_encoder,
            self.pipeline.text_encoder_2,
            self.pipeline.text_encoder_3,
        ):
            net.eval()
            net.requires_grad_(False)
            net.to(device, dtype)

        # default image size is 1024x1024 which give a latent size of 128
        self.n_channels = self.transformer.config.in_channels
        self.latent_size = self.pipeline.default_sample_size
        self.latent_shape = (self.n_channels, self.latent_size, self.latent_size)
        self.im_size = self.latent_size * self.pipeline.vae_scale_factor

        self.set_im_size(768)

        # these
        self.ctx = None
        self.prompt_embeds = None
        self.negative_prompt_embeds = None
        self.pooled_prompt_embeds = None
        self.negative_pooled_prompt_embeds = None
        self.use_cfg = True
        self.guidance = 7.0

        # empty any cache
        torch.cuda.empty_cache()

    def set_prompt(
        self,
        ctx: List[str],
        use_cfg: bool = True,
        guidance: float = 7.0,
        n_images_per_prompt: int = 1,
    ):
        if isinstance(ctx, str):
            ctx = [ctx]
        # BUG the list built by hydra throws an error when passing in it
        #   to the prompt encoder
        else:
            ctx = list(ctx)

        negative_ctx = [""] * len(ctx)
        (
            self.prompt_embeds,
            self.negative_prompt_embeds,
            self.pooled_prompt_embeds,
            self.negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=ctx,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_ctx,
            do_classifier_free_guidance=use_cfg,
            device=self.device,
            num_images_per_prompt=n_images_per_prompt,
            max_sequence_length=256,  # hardcoded in SD3
        )

        self.ctx = ctx
        self.use_cfg = use_cfg
        self.guidance = guidance

        if use_cfg:
            self.prompt_embeds = torch.cat(
                [self.negative_prompt_embeds, self.prompt_embeds], dim=0
            )
            self.pooled_prompt_embeds = torch.cat(
                [self.negative_pooled_prompt_embeds, self.pooled_prompt_embeds], dim=0
            )

    def set_im_size(self, im_size: int):
        """Image size should be a multiple of 8.

        Ideally, the image size should be between 512 and 1024
        """
        self.latent_size = im_size // self.pipeline.vae_scale_factor
        self.latent_shape = (self.n_channels, self.latent_size, self.latent_size)
        self.im_size = im_size

    def pred_velocity(self, x: Tensor, t) -> Tensor:
        """Requires `set_prompt()` to be called first."""
        if self.prompt_embeds is None or self.pooled_prompt_embeds is None:
            raise RuntimeError("You must call set_prompt() before calling forward().")

        # cast the input to the wrapper dtype
        x_dtype = x.dtype
        x = x.to(self.dtype)

        n_samples = len(x)
        # update cached prompt embeddings if needed
        # to account for the number of samples to be generated
        # NOTE: it won't change the cached embeddings if the prompt has changed
        self._update_cached_prompt_embs(x)

        if isinstance(t, int):
            t = torch.tensor([t], dtype=torch.int32, device=self.device)
        elif t.ndim in (0, 1):
            t = t.reshape(1)
        else:
            raise ValueError("`t` must be int or Tensor (either scaler or with dim=1).")

        net_timestep = self.net_timesteps.index_select(0, t.view(-1))

        if self.use_cfg:
            x = torch.cat([x] * 2)
            net_timestep = torch.cat([net_timestep] * 2 * n_samples)

        v_pred = self.transformer(
            hidden_states=x,
            timestep=net_timestep,
            encoder_hidden_states=self.prompt_embeds,
            pooled_projections=self.pooled_prompt_embeds,
            return_dict=False,
        )[0]

        if self.use_cfg:
            v_uncond, v_cond = v_pred.chunk(2)
            v_pred = self.guidance * v_cond + (1 - self.guidance) * v_uncond

        return v_pred.to(x_dtype)

    def decode(self, x: Tensor) -> Tensor:
        x_dtype = x.dtype

        latents = (x / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        # cast the input to the wrapper dtype
        latents = latents.to(self.dtype)
        image = self.vae.decode(latents, return_dict=False)[0]

        return image.to(x_dtype)

    def encode(self, x: Tensor) -> Tensor:
        # cast the input to the wrapper dtype
        x_dtype = x.dtype
        x = x.to(self.dtype)

        if isinstance(self.vae, AutoencoderTiny):
            z = self.vae.encode(x, return_dict=False)[0]
        else:
            latent_output = self.vae.encode(x)

            if hasattr(latent_output, "latent_dist"):
                z = latent_output.latent_dist.sample()
            else:
                z = latent_output.sample()

        z = z.to(x_dtype)
        latents = (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return latents

    def _update_cached_prompt_embs(self, x) -> None:
        n_samples = len(x)
        n_samples_per_prompt, r = divmod(n_samples, len(self.ctx))

        if r != 0:
            raise ValueError(
                "Number of samples should be a multiple of number of prompts\n"
                f"{n_samples_per_prompt=}, {n_samples=}"
            )

        # update cached prompts if needed
        coef = 2 if self.use_cfg else 1
        if coef * n_samples != len(self.prompt_embeds):
            self.set_prompt(
                self.ctx,
                use_cfg=self.use_cfg,
                guidance=self.guidance,
                n_images_per_prompt=n_samples_per_prompt,
            )


class SD3LatentWrapper(SD3Wrapper):
    """Wrapper for Latent inpainting."""

    # set encoder/decoder to the identity to work in latent space
    def decode(self, x) -> Tensor:
        return x

    def encode(self, x) -> Tensor:
        return x

    # methods to apply `real` encoder/decoder if needed
    def pixel_decode(self, x):
        return super().decode(x)

    def pixel_encode(self, x):
        return super().encode(x)
