# %%
import re
import json
import shutil
from pathlib import Path

import torch
import numpy as np

# ---
bench_path = Path("Provide the Path of the benchmark")
save_dir = Path("Provide save path")
# ---

# load mapping
with open(bench_path / "mapping_file.json", "r") as f:
    mappings = json.load(f)


# NOTE copy/paste of source code PnPInversion
# https://github.com/cure-lab/PnPInversion/blob/07f97f448150e2ca220bebd54c8f687c5c50c67a/run_editing_blended_latent_diffusion.py#L22-L39
def mask_decode(encoded_mask, image_shape=[512, 512]):
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))

    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i + 1], length - encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i] + j] = 1

    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1

    return mask_array


d_prompt_image = {}

for name, task in mappings.items():
    encoded_mask = task["mask"]
    mask = mask_decode(encoded_mask)

    # flip 0s and 1s as we use the convention 0: not visible and 1 for visible
    mask = 1 - mask
    # skip masks that cover the entire image
    if np.sum(mask) == 0.0:
        continue

    mask = torch.from_numpy(mask).to(torch.int8)
    mask = mask.reshape(1, 1, 512, 512)

    # get prompt and remove special characters
    prompt = re.sub(r"[\[\]]", "", task["editing_prompt"])
    d_prompt_image[name] = prompt

    # save mask
    torch.save(mask, save_dir / "masks" / f"{name}.pt")

    # cp images to different location
    src = bench_path / "annotation_images" / task["image_path"]
    ext = task["image_path"].split(".")[-1]
    dst = save_dir / "images" / f"{name}.{ext}"
    shutil.copyfile(src, dst)

# save prompts of each image
with open(save_dir / "prompts.json", "w") as f:
    json.dump(d_prompt_image, f, indent=4)
