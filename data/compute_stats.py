# %%
import torch
from metrics import FID, patchwiseFID

from pathlib import Path

# ---
path_imgs = "Provode path of folder containing images"
save_folder = Path("Provide save folder")
# ---

target_sizes = (512, 768, 1024)


device = "cuda:1"


# %%
torch.manual_seed(0)

d_latents = {}

for target_size in target_sizes:
    pfid = FID(
        device=device,
        target_size=target_size,
    )

    mean, cov = pfid._compute_stats(path_imgs)

    d_latents[f"{target_size}"] = {"mean": mean, "cov": cov}


# %%
save_path = save_folder / "5k_stats.pkl"

import pickle

with open(save_path, "wb") as f:
    pickle.dump(d_latents, f)


# %%
torch.manual_seed(0)

d_latents = {}

for target_size in target_sizes:
    pfid = patchwiseFID(
        n_patch_per_img=10,
        device=device,
        batch_size=100,
        target_size=target_size,
    )

    mean, cov = pfid._compute_stats(path_imgs)

    d_latents[f"{target_size}"] = {"mean": mean, "cov": cov}

# %%
save_path = save_folder / "5k_10_patch_256_stats.pkl"

import pickle

with open(save_path, "wb") as f:
    pickle.dump(d_latents, f)
