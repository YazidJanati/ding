## Dataset Details

This directory contains notes and helper scripts for preparing the datasets used in the project.

## FFHQ

The FFHQ 1024x1024 dataset was downloaded from the official repository:
[NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset).

Five subfolders were downloaded:

- `00000`
- `01000`
- `02000`
- `03000`
- `04000`

Together, these folders contain 5,000 images.

Merge the images from these five folders into:

```text
ffhq/images
```

## DIV2K

The DIV2K dataset was downloaded from the official website:
[DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

It contains two splits:

- train: the first 800 images
- validation: the last 100 images

Merge both splits into:

```text
DIV2K/images
```

Image captions were generated with BLIP-2 Flan-T5-XL using the following settings:

- maximum caption length: 100 tokens
- repetition penalty: 1.5

Before generating captions, download the BLIP-2 Flan-T5-XL model:

```bash
hf download Salesforce/blip2-flan-t5-xl
```

Then run the provided script:

```bash
python generate_captions_div2k.py
```

Update the paths at the beginning of `generate_captions_div2k.py` before running it.

## PIE-Bench

PIE-Bench was downloaded from:
[cure-lab/PnPInversion](https://github.com/cure-lab/PnPInversion).

It contains 700 images, each with:

- an original prompt
- an editing prompt
- a mask for the region to edit

The preprocessing step extracts:

- images
- masks
- prompts

Each extracted item is named after the corresponding image.

For some images, such as `924000000009`, the provided mask covers the entire image. These cases were removed from the benchmark. In total, 144 images were removed, leaving 556 images.

Run the provided preprocessing script:

```bash
python prepare_pie_bench.py
```

Update the paths at the beginning of `prepare_pie_bench.py` before running it.

Among other steps, this script:

- isolates the benchmark images in a separate folder
- creates the masks and saves them separately

## Precomputed Statistics for FID and Patchwise FID

To precompute and save statistics for FID and patchwise FID, use the provided script:

```bash
python compute_stats.py
```

Update the paths at the beginning of `compute_stats.py` before running it.
