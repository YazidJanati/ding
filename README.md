# Efficient Zero-Shot Inpainting with Decoupled Diffusion Guidance

This repository contains the DING algorithm for fast, training-free image inpainting with pre-trained diffusion models.

Paper: [Efficient Zero-Shot Inpainting with Decoupled Diffusion Guidance](https://arxiv.org/abs/2512.18365)


**NOTE:** We have released a modular python package that extends this work to additional models and datasets, see the [project webpage](https://ahmedgh970.github.io/ding-editor/) and package [ding-editor](https://github.com/ahmedgh970/ding-editor).


## Setup

1. Install the package in editable mode:

```bash
pip install -e .
```

Dependency details are available in [`pyproject.toml`](pyproject.toml).

2. Create `src/local_paths.py` and set the absolute path to this repository. You can use `pwd` to get the path:

```python
from pathlib import Path

REPO_PATH = Path("/path/to/repository")  # Replace with the absolute path to this repository.
```

## Downloading Models

Download the Stable Diffusion 3 models:

```bash
# Stable Diffusion 3.5 Medium
hf download stabilityai/stable-diffusion-3.5-medium

# Stable Diffusion 3 Medium
hf download stabilityai/stable-diffusion-3-medium-diffusers
```

Download the inpainting ControlNet:

```bash
hf download alimama-creative/SD3-Controlnet-Inpainting
```

## Datasets

Dataset details are provided in the [`data`](data) folder, including:

- dataset descriptions
- preprocessing steps and scripts

## Running Demos

Run the default demo with:

```bash
python3 runner.py
```

The script behavior can be adapted by changing the hyperparameters in [`config/runner.yaml`](config/runner.yaml).

Run the demo with a custom image and prompt:

```bash
python3 runner.py \
  im_abs_path=assets/0847.png \
  conditioning.ctx="A beautifully preserved vintage black car with the license plate 'AX 7681' parked on a charming street, in front of a shop decorated with Welsh dragon flags and a Dutch flag."
```

Run the demo with a custom mask:

```bash
python3 runner.py \
  im_abs_path=assets/000000000001.jpg \
  conditioning.ctx="a square cake with orange frosting on a wooden plate" \
  task.path_mask=assets/000000000001.pt
```

## Evaluation

The `runner.py` script reports per-sample metrics, including:

- LPIPS
- cPSNR
- runtime
- GPU memory consumption

To compute CLIP-Score:

1. Download the CLIP model:

```bash
hf download openai/clip-vit-large-patch14
```

2. Run the script with `compute_clip=True`.

To compute distribution-level metrics, run `eval.py`:

```bash
python3 eval.py \
  eval.path_ref_stats_pfid="path/to/ref/stats/pfid" \
  eval.path_ref_stats_fid="path/to/ref/stats/fid"
```


## Citation

If you find this work useful, please cite:

```bibtex
@article{moufad2026ding,
  title={Efficient Zero-Shot Inpainting with Decoupled Diffusion Guidance},
  author={Moufad, Badr and Shouraki, Navid Bagheri and
          Durmus, Alain Oliviero and Hirtz, Thomas and
          Moulines, Eric and Olsson, Jimmy and Janati, Yazid},
  journal={ICLR 2026},
  year={2026}
}

@article{ghorbel2026ding-editor,
  title={When Test-Time Guidance Is Enough:
         Fast Image and Video Editing with Diffusion Guidance},
  author={Ghorbel, Ahmed and Moufad, Badr and Shouraki, Navid Bagheri
          and Durmus, Alain Oliviero and Hirtz, Thomas and
          Moulines, Eric and Olsson, Jimmy and Janati, Yazid},
  journal={ICLR 2026, ReALM-GEN Workshop},
  year={2026}
}
```
