## Details about the dataset

**About FFHQ**

FFHQ 1024x1024 dataset downloaded from https://github.com/NVlabs/ffhq-dataset

Five subfolders were download: 
    00000, 01000, 02000, 03000, and 04000,
which accounts for 5k images

- Merge the images in the five folders into on ``ffhq/ìmages``


**About DIV2K**

DIV2K dataset downloaded from the official website https://data.vision.ee.ethz.ch/cvl/DIV2K/

It comprises two partition
- train: first 800 images
- valiation: last 100 images

- Merge The two partition into one ``DIV2K/images``.

The captions of the images were generated using BLIP-2 flan-t5-xl with a max token of the caption equal 100 and repetition_penalty equals 1.5

Use the provided scripts ``generate_captions_div2k.py`` and modify the paths accordingly in the beginning of the scripts.

Beforehand, download the BLIP-2 flan-t5-xl model
```bash
hf download Salesforce/blip2-flan-t5-xl
```

**About PIE-Bench**

PIE-Bench was downloaded from https://github.com/cure-lab/PnPInversion

It comprises 700 images with original prompt, editing prompt, and mask for the part to be edited

The applied preprocessing comprises isolating
    - images
    - masks
    - prompts
the name of each is the name of the image.

For some images, e.g. image "924000000009", the provided mask cover the entire image.
These, which account for 144 of the Bench, were removed which leave 556 images.

Use the provided scripts ``prepare_pie_bench.py`` and modify the paths accordingly in the beginning of the scripts.

Among others, the scripts will
- isolate the images for the benchmark in a separate folder
- create the mask and save them separately


## Generate precomputed statistics for FID, and patchwise FID

To pre-compute and save statistics for FID and patch FID, use the provided scripts ``compute_stats.py`` and modify the paths accordingly in the beginning of the scripts
