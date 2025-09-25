import PIL
import torch
import numpy as np

import matplotlib.pyplot as plt


def display(x, save_path=None, title=None):
    sample = x.squeeze(0).float().permute(1, 2, 0)
    sample = (sample + 1.0) * 127.5
    sample = sample.squeeze()
    sample = sample.cpu().numpy().astype(np.uint8)
    img_pil = PIL.Image.fromarray(sample)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img_pil)
    if title:
        ax.set_title(title)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.show()
    if save_path is not None:
        fig.savefig(save_path + ".png")


def get_pil_image(x: torch.Tensor):
    if x.dtype != torch.uint8:
        x = x.clamp(-1.0, 1.0)
        x = (x + 1.0) * 127.5

    sample = x.cpu().permute(0, 2, 3, 1)

    sample = sample.numpy().astype(np.uint8)
    if sample.shape[0] == 1:
        img_pil = PIL.Image.fromarray(sample[0])
        return img_pil
    else:
        return [PIL.Image.fromarray(s) for s in sample]


def check_image(tensor):
    assert (
        torch.max(tensor) <= 1.0 and torch.min(tensor) >= -1.0
    ), "Output images should be (-1, 1.)"


def normalize_tensor(tensor):
    check_image(tensor)
    return (tensor + 1.0) / 2.0
