import os
import json
import time
from pathlib import Path
import hydra
from omegaconf import DictConfig

import torch

from fm.wrappers import AVAILABLE_MODELS
from fm.samplers import AVAILABLE_SAMPLERS
from fm.operators import BoxInpainting
from fm.inv_problem import create_inv_prob, load_img
from fm.wrappers import NetCFM

from image_utils import get_pil_image
from metrics import LPIPS, PSNR, MaskedPSNR, ClipScore
from experiments_tools import (
    fix_seed,
    get_gpu_memory_consumption,
    update_sampler_cfg,
)

from local_paths import REPO_PATH


dtype = torch.float32
model_dtype = torch.bfloat16


@hydra.main(config_path=str(REPO_PATH / "configs"), config_name="run")
def run_sampler(cfg: DictConfig):

    fix_seed(cfg.seed)
    torch.set_default_device(cfg.device)
    torch.set_default_dtype(dtype)
    torch.cuda.empty_cache()

    # set paths
    save_folder = Path(cfg.save_dir)

    # these folder should be created beforehand
    imgs_save_dir = save_folder / "reconstructions"
    metric_save_dir = save_folder / "metrics"

    imgs_save_dir.mkdir(exist_ok=True, parents=True)
    metric_save_dir.mkdir(exist_ok=True, parents=True)
    # ---

    # update algorithm hyperparameters based on the task
    if cfg.use_context_parameters:
        update_sampler_cfg(cfg, context_fields=("task_name",))

    # init model
    model: NetCFM = AVAILABLE_MODELS[cfg.model.name](
        device=cfg.device, dtype=model_dtype, **cfg.model.parameters
    )

    # set-up model
    model.set_im_size(cfg.target_size)

    # init clip model for scoring
    # to avoid side-effects, don't init model unless specified
    if cfg.compute_clip:
        clip = ClipScore(device=cfg.device)

    # path of the img
    img_path = cfg.im_abs_path
    img_name = img_path.split("/")[-1].split(".")[-1]

    # set context
    model.set_prompt(**cfg.conditioning)

    ref_img = load_img(
        img=img_path,
        device=cfg.device,
        dtype=dtype,
        target_size=cfg.target_size,
    )

    is_latent_op = getattr(cfg.task, "is_latent", False)
    img = (
        model.pixel_encode(ref_img.unsqueeze(0).to(cfg.device)).squeeze(0)
        if is_latent_op
        else ref_img
    )

    # update mask in case of PIE-bench full run
    if cfg.task_name == "mask_inpainting" and cfg.path_dir_masks not in ("", None):
        cfg.task.path_mask = Path(cfg.path_dir_masks) / f"{img_name}.pt"

    # create inverse problem
    inv_prob = create_inv_prob(
        img=img,
        device=cfg.device,
        dtype=dtype,
        obs_std=cfg.obs_std,
        **cfg.task,
    )

    # solve problem
    initial_noise = torch.randn((cfg.n_samples, *model.latent_shape))
    sampler = AVAILABLE_SAMPLERS[cfg.sampler.name]
    # HACK controlnet uses reference images which will break when the operator is latent
    # as we define x_ref in this case as encoding of the imag
    if cfg.sampler.name == "alimama_controlnet":
        inv_prob.x_ref = ref_img[None].to(cfg.device, dtype)

    start_time = time.perf_counter()
    samples = sampler(model, inv_prob, initial_noise, **cfg.sampler.parameters)

    # decode in case of latent inpainting
    if is_latent_op:
        samples = model.pixel_decode(samples)

    end_time = time.perf_counter()

    samples = samples.clamp(-1, 1)

    # save reconstructions
    imgs = get_pil_image(samples)

    # NOTE handle this case as get_pil_image return the a PIL images
    # instead of a list of images when the number of samples equals 1
    if cfg.n_samples == 1:
        imgs.save(imgs_save_dir / f"rec-{img_name}.png")
    else:
        for idx, im in enumerate(imgs):
            im.save(imgs_save_dir / f"rec-{img_name}_{idx}.png")

    # Compute metrics
    x_ref = ref_img.to(cfg.device)
    lpips, psnr = LPIPS(), PSNR()

    lpips_score = lpips.score(samples, x_ref)
    psnr_score = psnr.score(samples, x_ref)

    # save metrics
    best_idx = psnr_score.argmax()
    lpips_score, psnr_score = (
        lpips_score[best_idx].item(),
        psnr_score[best_idx].item(),
    )
    results_dic = {
        "lpips": lpips_score,
        "psnr": psnr_score,
    }

    # add pnsr for the observed part in the case inpainting
    if isinstance(inv_prob.H_func, BoxInpainting):
        mask = inv_prob.H_func.mask_3d.float()
        # resize mask if task is in latent
        if is_latent_op:
            mask = torch.nn.functional.interpolate(
                mask, size=cfg.target_size, mode="nearest"
            )

        masked_pnsr = MaskedPSNR(mask)
        masked_psnr_score = masked_pnsr.score(samples, x_ref)

        results_dic["psnr_obs"] = masked_psnr_score[best_idx].item()

        # to save observation
        obs_img = get_pil_image((mask * x_ref[None]).clamp(-1, 1))
        obs_img.save(imgs_save_dir / f"obs-{img_name}.png")

    # compute clip scores
    if cfg.compute_clip:
        clip_score = clip.compute_score(samples, cfg.conditioning.ctx)
        results_dic["clip"] = clip_score[best_idx].item()

        if isinstance(inv_prob.H_func, BoxInpainting):
            # a mask that keeps the edited (inpainted) part (convert 0s to 1s and vice-versa)
            flipped_mask = (1 - mask).clamp(-1, 1)
            clip_ed_score = clip.compute_score(
                flipped_mask * samples, cfg.conditioning.ctx
            )
        results_dic["clip_ed"] = clip_ed_score[best_idx].item()

    # runtime and gpu consumption
    runtime = end_time - start_time
    gpu_consuption = get_gpu_memory_consumption(cfg.device)

    results_dic.update({"runtime": runtime, "gpu_consumption": gpu_consuption})

    # save metrics
    with open(metric_save_dir / f"{img_name}.json", "w") as f:
        json.dump(results_dic, f, indent=4)

    print(f"Finished {cfg.task_name}-{cfg.sampler.name}-img-{img_name} with steps")


if __name__ == "__main__":
    run_sampler()
