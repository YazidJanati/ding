import os
import json
from tqdm import tqdm
from pathlib import Path

import torch
from fm.inv_problem import load_img

device, dtype = "cuda:0", torch.float16

# ---
save_path = Path("Provide save path")
p_data = Path("Provide path to  DIV2K images")
# ---


# Load model + processor
from transformers import Blip2Processor, Blip2ForConditionalGeneration


processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    torch_dtype=torch.float16,  # use half precision for speed/memory
).to(device)

model.to(device)
model.requires_grad_(False)


d_caption_image = {}

images_names = sorted(os.listdir(p_data))

for img_name in tqdm(images_names):
    name = img_name.split(".")[0]

    # load image
    im_abs_path = p_data / img_name
    img = load_img(im_abs_path, target_size=768, device=device, dtype=dtype)

    # preprocess image
    img_input = (img + 1) / 2
    inputs = processor(
        img_input,
        ["Provide a single concise description of this image."],
        do_rescale=False,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Generate caption
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            repetition_penalty=1.5,
        )
    caption = processor.batch_decode(output, skip_special_tokens=True)

    # save caption
    d_caption_image[name] = caption[0]


with open(save_path, "w") as f:
    json.dump(d_caption_image, f, indent=4)
