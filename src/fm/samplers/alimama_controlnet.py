from torch import Tensor

from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetModel
from diffusers.pipelines import StableDiffusion3ControlNetInpaintingPipeline

from fm.wrappers import SD3Wrapper
from fm.inv_problem import InverseProblem


def alimama_controlnet_sampler(
    dm: SD3Wrapper,
    inv_problem: InverseProblem,
    initial_noise: Tensor,
    n_steps: int = 28,
    guidance_scale: float = 7.0,
    controlnet_conditioning_scale: float = 0.95,
    control_net_path: str = None,
):
    """Wrapper for Alimama Control-Net.

    Adapted from https://huggingface.co/alimama-creative/SD3-Controlnet-Inpainting
    and source code

    Notes
    -----
    - The control-net is for Stable Diffusion 3 Medium (though, it works for SD3.5 medium)
    """

    # load controlnet
    # NOTE loading does not take a lot of time (~1 second)
    # NOTE here, the preloaded controlnet is provided otherwise, it will be accounted for in the runtime
    controlnet = dm.controlnet
    if controlnet is None:
        controlnet = SD3ControlNetModel.from_pretrained(
            control_net_path,
            use_safetensors=True,
            extra_conditioning_channels=1,
        ).to(dm.device, dm.dtype)

    # init pipeline + controlnet
    pipe = StableDiffusion3ControlNetInpaintingPipeline(
        dm.pipeline.transformer,
        dm.pipeline.scheduler,
        dm.pipeline.vae,
        dm.pipeline.text_encoder,
        dm.pipeline.tokenizer,
        dm.pipeline.text_encoder_2,
        dm.pipeline.tokenizer_2,
        dm.pipeline.text_encoder_3,
        dm.pipeline.tokenizer_3,
        # ---
        controlnet,
    )

    n_samples = initial_noise.shape[0]
    num_images_per_prompt = n_samples // len(dm.ctx)
    width, height = dm.im_size, dm.im_size

    # NOTE upon inspection of control-Net
    #   - image: the scale [-1, 1] is adopted
    #   - mask: 1 for masked pixels 0 otherwise
    image = inv_problem.x_ref
    mask = (1 - inv_problem.H_func.pixel_mask).clamp(0.0, 1.0)

    # NOTE negative prompt is used in their provided code
    latents = pipe(
        negative_prompt="deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW",
        prompt=dm.ctx,
        height=height,
        width=width,
        control_image=image,
        control_mask=mask,
        num_inference_steps=n_steps,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        # by default 7, 5 work also under 4 perfs start to drop
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        output_type="latent",
    )[0]

    latents = latents.to(initial_noise.dtype)
    return dm.decode(latents)
