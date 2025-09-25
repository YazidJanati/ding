# this scripts performs
#     - merging of runs on each image
#     - save a summary of the run
#           * mean metrics
#           * configs of the run
#

import os
import json
from pathlib import Path
from datetime import datetime

import hydra
import torch
import pandas as pd
from omegaconf import OmegaConf, DictConfig

from metrics import patchwiseFID, FID
from experiments_tools import update_sampler_cfg
from local_paths import REPO_PATH


@hydra.main(config_path=str(REPO_PATH / "configs/fm"), config_name="exp_jz_fm")
def run_experiment(cfg: DictConfig):
    torch.set_default_device(cfg.device)

    save_folder = Path(cfg.save_dir)
    metric_save_dir = save_folder / "metrics"
    imgs_save_dir = save_folder / "reconstructions"

    # update algorithm hyperparameters based on the task
    if cfg.use_context_parameters:
        update_sampler_cfg(cfg, context_fields=("task_name",))

    # compute patchwise FID
    pfid = patchwiseFID(
        path_ref_stats=cfg.eval.path_ref_stats_pfid,
        batch_size=cfg.eval.batch_size,
        n_patch_per_img=cfg.eval.n_patch_per_img,
        patch_size=cfg.eval.patch_size,
        device=cfg.device,
        target_size=cfg.target_size,
        # BUG: sometimes raises an error with CUDA when not set to zero
        # potentially there a poblem with multiprocessing
        num_workers=0,
    )
    pfid_score = pfid.compute_patched_FID(imgs_save_dir)

    # regular FID
    fid = FID(
        path_ref_stats=cfg.eval.path_ref_stats_fid,
        batch_size=cfg.eval.batch_size,
        device=cfg.device,
        target_size=cfg.target_size,
        # BUG: sometimes raises an error with CUDA when not set to zero
        # potentially there a poblem with multiprocessing
        num_workers=0,
    )
    fid_score = fid.compute_FID(imgs_save_dir)

    print("Running checkout script")

    # details
    details = {}
    for j_path in os.listdir(metric_save_dir):
        idx_img = j_path.replace(".json", "")

        with open(metric_save_dir / j_path) as f:
            content = json.load(f)

        details[idx_img] = content

    print("Individual runs metrics merged")

    # run setup
    results_df = pd.DataFrame.from_dict(details, orient="index")
    log_data = {
        "model": cfg.model.name,
        "task": cfg.task_name,
        "sampler_name": cfg.sampler.name,
        "n_images": len(os.listdir(imgs_save_dir)),
        "timestamp": datetime.now().strftime(r"%Y%m%d_%H%M%S"),
        "pfid": pfid_score,
        "fid": fid_score,
        "results_mean": results_df.mean().to_dict(),
        "results_median": results_df.median().to_dict(),
        "results_std": results_df.std().to_dict(),
        "job_cfg": OmegaConf.to_container(cfg, resolve=True),
    }

    print("Run steps saved")

    # save all
    merged_results = {
        "log_data": log_data,
        "details": details,
    }
    with open(save_folder / "all_metrics.json", "w") as f:
        json.dump(merged_results, f)


if __name__ == "__main__":
    run_experiment()
