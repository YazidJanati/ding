# Efficient Zero-shot Inpainting with Decoupled Diffusion Guidance

Ding algorithm for fast training-free inpainting using pre-trained diffusion models.

## Setups the repository

Add the the following two files to

1. Install the code in editable mode.
```bash
pip install -e .
```

Details about the dependencies can be in ``pyproject.toml``

2. then put the in ``src/local_paths.py`` file, the absolute paths to the project (hint use the command ``pwd``)

```python
from pathlib import Path

REPO_PATH = Path("/path/to/repository")     # <--- change it with the absolute path of the folder
```


## Downloading models

- For downloading SD3 models run

```bash
# for 3.5 medium
hf download stabilityai/stable-diffusion-3.5-medium

# for 3 medium
hf download stabilityai/stable-diffusion-3-medium-diffusers
```

- To download the Inpainting Controlnet, run
```bash
hf download alimama-creative/SD3-Controlnet-Inpainting
```

## Datasets

We provide details about the dataset in the folder ``data``, namely
- description of dataset
- preprocessing steps and scripts


## Running demos

The can be run using the script, by
```bash
python3 runner.py
```

the behavior of the script can be adapted by changing the hyperparmeters in configuration files ``config/runner.yaml``.

Running code with different image/prompt
```bash
python3 runner.py \
  im_abs_path=assets/0847.png\
  conditioning.ctx="A beautifully preserved vintage black car with the license plate 'AX 7681' parked on a charming street, in front of a shop decorated with Welsh dragon flags and a Dutch flag."
```

Providing a custom mask
```bash
python3 runner.py \
  im_abs_path=assets/000000000001.jpg \
  conditioning.ctx="a square cake with orange frosting on a wooden plate"
  task.path_mask=assets/000000000001.pt
```


## Evaluation

The ``runner.py`` script provides per sample metrics, namely, 
  - LPIPS
  - cPSNR
  - runtime
  - GPU memory consumption

For CLIP-Score, however ensure,

1. Download the CLIP model using
```bash
hf download openai/clip-vit-large-patch14
```

2. and run the script with ``compute_clip=True``

To get distribution-wise metrics, run ``eval.py``
```bash
python3 eval.py \
  eval.path_ref_stats_pfid="path/to/ref/stats/fid" \
  eval.path_ref_stats_fid="path/to/ref/stats/fid"
```
