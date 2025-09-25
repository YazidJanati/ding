import pickle

from tqdm import tqdm
from typing import Tuple
from pathlib import Path

import numpy as np

import torch
from torch import Tensor
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.transforms import v2 as tv_transforms
from torchmetrics.functional.multimodal.clip_score import _clip_score_update

from transformers import CLIPModel as _CLIPModel
from transformers import CLIPProcessor as _CLIPProcessor

import lpips
from image_utils import normalize_tensor
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import (
    compute_statistics_of_path,
    calculate_frechet_distance,
    IMAGE_EXTENSIONS,
    ImagePathDataset,
)


class PSNR:
    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def score(self, samples: torch.Tensor, references: torch.Tensor):
        # samples: B, C, H, W
        # references: 1, C, H, W or B, C, H, W
        B = samples.shape[0]
        samples = normalize_tensor(samples)
        references = normalize_tensor(references)
        if references.shape[0] == 1:
            references = references.repeat(B, 1, 1, 1)

        mse = torch.mean((samples - references) ** 2, dim=(1, 2, 3))
        peak = 1.0  # we normalize the image to (0., 1.)
        psnr = 10 * torch.log10(peak / mse)
        return psnr.detach().cpu()


class MaskedPSNR:
    """For inpainting computes the PSNR for the observed part."""

    def __init__(self, mask: torch.Tensor):
        # mask must have shape (1, 1, height, width)
        self.mask = mask

    @torch.no_grad()
    def score(self, samples: torch.Tensor, references: torch.Tensor):
        # samples: B, C, H, W
        # references: 1, C, H, W or B, C, H, W
        B = samples.shape[0]
        samples = normalize_tensor(samples)
        references = normalize_tensor(references)
        if references.shape[0] == 1:
            references = references.repeat(B, 1, 1, 1)

        masked_diff = self.mask * (samples - references)
        mse = torch.sum(masked_diff**2, dim=(1, 2, 3))
        # NOTE mask consistent of 0s and 1s where the 1s appear in the visible pixel
        # hence summing all elements of the mask equal the number of visible pixels
        n_pixels = self.mask.sum()
        mse /= n_pixels

        peak = 1.0  # we normalize the image to (0., 1.)
        psnr = -10 * torch.log10(mse / peak)
        return psnr.cpu()


def check_device(tensor, device):
    if tensor.device != device:
        tensor = tensor.to(device)
    return tensor


def check_image(tensor):
    assert torch.max(tensor) <= 1.0 + 1e-3 and torch.min(tensor) >= -1.0 - 1e-3


class LPIPS:
    def __init__(self, base_model="alex", device="cpu") -> None:
        self.device = device
        self.loss_fn = lpips.LPIPS(net=base_model).to(device)

    @torch.no_grad()
    def score(self, samples: torch.Tensor, references: torch.Tensor):
        # ! Notice that samples and references should be in [-1, 1]
        check_image(samples)
        check_image(references)
        samples = check_device(samples, self.device)
        references = check_device(references, self.device)
        return self.loss_fn(samples, references).detach().cpu()

    def on_dir():
        pass


class patchwiseFID:
    def __init__(
        self,
        path_ref_stats: str | Path = None,
        path_ref_imgs: str | Path = None,
        batch_size: int = 50,
        dims: int = 2048,
        n_patch_per_img: int = 5,
        patch_size: int = 256,
        num_workers: int = 1,
        device: str = "cpu",
        target_size: int = None,
    ):
        # NOTE loading reference statistics is prioritized
        self.path_ref_stats = path_ref_stats
        self.path_ref_imgs = path_ref_imgs

        self.batch_size = batch_size
        self.dims = dims
        self.num_workers = num_workers
        self.device = device

        transformations = []
        # if provided, resize the images to ensure ref and generated images
        # have the same resolution when computing the FID
        self.target_size = target_size
        if target_size is not None:
            transformations = [
                tv_transforms.Resize(target_size),
                tv_transforms.CenterCrop(target_size),
            ]
        # convert to tensor
        transformations += [
            tv_transforms.ToImage(),
            tv_transforms.ToDtype(torch.float32, scale=True),
        ]
        self.img_preprocessor = tv_transforms.Compose(transformations)

        self.patch_size = patch_size
        self.n_patch_per_img = n_patch_per_img
        self.random_crop = tv_transforms.RandomCrop(size=(patch_size, patch_size))

        # load Inception V3 model
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(device)

        # stats of real data
        if path_ref_stats is not None:
            with open(path_ref_stats, "rb") as f:
                ref_metrics = pickle.load(f)
                ref_metrics = ref_metrics[f"{target_size}"]

            self.mean_ref, self.cov_ref = ref_metrics["mean"], ref_metrics["cov"]
        # else:
        #     self.mean_ref, self.cov_ref = self._compute_stats(path_ref_imgs)

    def compute_patched_FID(self, path_imgs: str | Path) -> float:
        # compute stat of data
        mean_gen, cov_gen = self._compute_stats(path_imgs)

        return calculate_frechet_distance(
            self.mean_ref, self.cov_ref, mean_gen, cov_gen
        )

    def _compute_stats(self, path_imgs: str) -> Tuple[Tensor, Tensor]:

        # NOTE this an adapted version of the function ``get_activations``
        # from ``pytorch_fid.fid_score``; the adapted version repeatedly
        # computes activations over random crop of the inputs

        # ---
        # load images
        path = Path(str(path_imgs))
        files = sorted(
            [file for ext in IMAGE_EXTENSIONS for file in path.glob("*.{}".format(ext))]
        )

        self.model.eval()
        model = self.model
        batch_size = self.batch_size

        if batch_size > len(files):
            print(
                (
                    "Warning: batch size is bigger than the data size. "
                    "Setting batch size to data size"
                )
            )
            batch_size = len(files)

        dataset = ImagePathDataset(files, transforms=self.img_preprocessor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

        # compute activations
        # NOTE account for the number of patch per image
        pred_arr = np.empty((len(files) * self.n_patch_per_img, self.dims))

        start_idx = 0

        for batch_imgs in tqdm(dataloader):
            batch_imgs = batch_imgs.to(self.device)

            for _ in range(self.n_patch_per_img):
                batch = self.random_crop(batch_imgs)

                with torch.no_grad():
                    pred = model(batch)[0]

                # If model output is not scalar, apply global spatial average pooling.
                # This happens if you choose a dimensionality not equal 2048.
                if pred.size(2) != 1 or pred.size(3) != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

                pred = pred.squeeze(3).squeeze(2).cpu().numpy()

                pred_arr[start_idx : start_idx + pred.shape[0]] = pred

                start_idx = start_idx + pred.shape[0]

        act = pred_arr
        # ---

        # compute mean cov
        mean = np.mean(act, axis=0)
        cov = np.cov(act, rowvar=False)

        return mean, cov


class FID:
    def __init__(
        self,
        path_ref_stats: str | Path = None,
        path_ref_imgs: str | Path = None,
        batch_size: int = 50,
        dims: int = 2048,
        num_workers: int = 1,
        device: str = "cpu",
        target_size: int = None,
    ):
        # NOTE loading reference statistics is prioritized
        self.path_ref_stats = path_ref_stats
        self.path_ref_imgs = path_ref_imgs

        self.batch_size = batch_size
        self.dims = dims
        self.num_workers = num_workers
        self.device = device

        transformations = []
        # if provided, resize the images to ensure ref and generated images
        # have the same resolution when computing the FID
        self.target_size = target_size
        if target_size is not None:
            transformations = [
                tv_transforms.Resize(target_size),
                tv_transforms.CenterCrop(target_size),
            ]
        # convert to tensor
        transformations += [
            tv_transforms.ToImage(),
            tv_transforms.ToDtype(torch.float32, scale=True),
        ]
        self.img_preprocessor = tv_transforms.Compose(transformations)

        # load Inception V3 model
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(device)

        # stats of real data
        if path_ref_stats is not None:
            with open(path_ref_stats, "rb") as f:
                ref_metrics = pickle.load(f)
                ref_metrics = ref_metrics[f"{target_size}"]

            self.mean_ref, self.cov_ref = ref_metrics["mean"], ref_metrics["cov"]
        # else:
        #     self.mean_ref, self.cov_ref = self._compute_stats(path_ref_imgs)

    def compute_FID(self, path_imgs: str | Path) -> float:
        # compute stat of data
        mean_gen, cov_gen = self._compute_stats(path_imgs)

        return calculate_frechet_distance(
            self.mean_ref, self.cov_ref, mean_gen, cov_gen
        )

    def _compute_stats(self, path_imgs: str) -> Tuple[Tensor, Tensor]:

        # NOTE this an adapted version of the function ``get_activations``
        # from ``pytorch_fid.fid_score``; to resize images before

        # ---
        # load images
        path = Path(str(path_imgs))
        files = sorted(
            [file for ext in IMAGE_EXTENSIONS for file in path.glob("*.{}".format(ext))]
        )

        self.model.eval()
        model = self.model
        batch_size = self.batch_size

        if batch_size > len(files):
            print(
                (
                    "Warning: batch size is bigger than the data size. "
                    "Setting batch size to data size"
                )
            )
            batch_size = len(files)

        dataset = ImagePathDataset(files, transforms=self.img_preprocessor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

        # compute activations
        # NOTE account for the number of patch per image
        pred_arr = np.empty((len(files), self.dims))

        start_idx = 0

        for batch_imgs in tqdm(dataloader):
            batch_imgs = batch_imgs.to(self.device)

            with torch.no_grad():
                pred = model(batch_imgs)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()

            pred_arr[start_idx : start_idx + pred.shape[0]] = pred

            start_idx = start_idx + pred.shape[0]

        act = pred_arr
        # ---

        # compute mean cov
        mean = np.mean(act, axis=0)
        cov = np.cov(act, rowvar=False)

        return mean, cov


class ClipScore:
    """Object to compute CLIP scores for a images and list of prompts.

    In case, the model can be downloaded using::
        hf download openai/clip-vit-large-patch14

    """

    def __init__(
        self, device: str = "cpu", model_name: str = "openai/clip-vit-large-patch14"
    ):

        model = _CLIPModel.from_pretrained(model_name)
        processor = _CLIPProcessor.from_pretrained(model_name)

        model.requires_grad_(False)
        model.to(device)

        self.model, self.processor = model, processor
        self.device = device

    def compute_score(self, imgs: Tensor, prompts: str):
        """imgs must be a tensor between [-1, 1]"""
        # NOTE HF clip accepts images as tensor with values [0, 255]
        imgs = (imgs + 1) / 2  # map to [0, 1]
        imgs = (imgs * 255).to(torch.uint8)  # map to [0, 255]

        # NOTE when there is one prompt, clip doesn't accept it as a list
        if len(prompts) == 1:
            prompts = prompts[0]

        score, _ = _clip_score_update(imgs, prompts, self.model, self.processor)
        score = torch.max(score, torch.zeros_like(score))

        return score.cpu()
