# import sys
# sys.path.append('./')

from typing import Tuple

import os
import cv2
import math
import torch
import random
import numpy as np
import argparse

import PIL
from PIL import Image

import diffusers
# from diffusers.utils import load_image
# from diffusers.models import ControlNetModel
# from diffusers import LCMScheduler

# from huggingface_hub import hf_hub_download
from style_template import styles
# from pipeline_stable_diffusion_xl_instantid_full import StableDiffusionXLInstantIDPipeline
from model_util import load_models_xl, get_torch_device, torch_gc

import gradio as gr


# global variable
MAX_SEED = np.iinfo(np.int32).max
# device = get_torch_device()
# dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Watercolor"

# # Load face encoder
# app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# app.prepare(ctx_id=0, det_size=(640, 640))

# Path to InstantID models
face_adapter = f'./checkpoints/ip-adapter.bin'
controlnet_path = f'./checkpoints/ControlNetModel'



def get_example():
        case = [
            [
                './examples/yann-lecun_resize.jpg',
                "a man",
                "Snow",
                "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
            ],
            [
                './examples/musk_resize.jpeg',
                "a man",
                "Mars",
                "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
            ],
            [
                './examples/sam_resize.png',
                "a man",
                "Jungle",
                "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, gree",
            ],
            [
                './examples/schmidhuber_resize.png',
                "a man",
                "Neon",
                "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
            ],
            [
                './examples/kaifu_resize.png',
                "a man",
                "Vibrant Color",
                "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
            ],
        ]
        return case

def run_for_examples(face_file, prompt, style, negative_prompt):
    return (face_file, None, prompt, negative_prompt, style, 30, 0.8, 0.8, 5, 42, False, True)



### Description
title = r"""
<h1 align="center">InstantID: Zero-shot Identity-Preserving Generation in Seconds</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://github.com/InstantID/InstantID' target='_blank'><b>InstantID: Zero-shot Identity-Preserving Generation in Seconds</b></a>.<br>

How to use:<br>
1. Upload an image with a face. For images with multiple faces, we will only detect the largest face. Ensure the face is not too small and is clearly visible without significant obstructions or blurring.
2. (Optional) You can upload another image as a reference for the face pose. If you don't, we will use the first detected face image to extract facial landmarks. If you use a cropped face at step 1, it is recommended to upload it to define a new face pose.
3. Enter a text prompt, as done in normal text-to-image models.
4. Click the <b>Submit</b> button to begin customization.
5. Share your customized photo with your friends and enjoy! 😊
"""

article = r"""
---
📝 **Citation**
<br>
If our work is helpful for your research or applications, please cite us via:
```bibtex
@article{wang2024instantid,
title={InstantID: Zero-shot Identity-Preserving Generation in Seconds},
author={Wang, Qixun and Bai, Xu and Wang, Haofan and Qin, Zekui and Chen, Anthony},
journal={arXiv preprint arXiv:2401.07519},
year={2024}
}
```
📧 **Contact**
<br>
If you have any questions, please feel free to open an issue or directly reach us out at <b>haofanwang.ai@gmail.com</b>.
"""

tips = r"""
### Usage tips of InstantID
1. If you're not satisfied with the similarity, try increasing the weight of "IdentityNet Strength" and "Adapter Strength."    
2. If you feel that the saturation is too high, first decrease the Adapter strength. If it remains too high, then decrease the IdentityNet strength.
3. If you find that text control is not as expected, decrease Adapter strength.
4. If you find that realistic style is not good enough, go for our Github repo and use a more realistic base model.
"""

css = '''
.gradio-container {width: 85% !important}
'''

with gr.Blocks(css=css) as demo:

    # description
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            
            # upload face image
            face_file = gr.Image(label="Upload a photo of your face", type="filepath")

            # optional: upload a reference pose image
            pose_file = gr.Image(label="Upload a reference pose image (optional)", type="filepath")
        
            # prompt
            prompt = gr.Textbox(label="Prompt",
                    info="Give simple prompt is enough to achieve good face fidelity",
                    placeholder="A photo of a person",
                    value="")
            
            submit = gr.Button("Submit", variant="primary")
            
            enable_LCM = gr.Checkbox(
                label="Enable Fast Inference with LCM",
                info="LCM speeds up the inference step, the trade-off is the quality of the generated image. It performs better with portrait face images rather than distant faces",
            )
            style = gr.Dropdown(label="Style template", choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME)
            
            # strength
            identitynet_strength_ratio = gr.Slider(
                label="IdentityNet strength (for fidelity)",
                minimum=0,
                maximum=1.5,
                step=0.05,
                value=0.80,
            )
            adapter_strength_ratio = gr.Slider(
                label="Image adapter strength (for detail)",
                minimum=0,
                maximum=1.5,
                step=0.05,
                value=0.80,
            )
            
            with gr.Accordion(open=False, label="Advanced Options"):
                negative_prompt = gr.Textbox(
                    label="Negative Prompt", 
                    placeholder="low quality",
                    value="(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
                )
                num_steps = gr.Slider( 
                    label="Number of sample steps",
                    minimum=20,
                    maximum=100,
                    step=1,
                    value=5
                )
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1,
                    value=0 
                )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=42,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                enhance_face_region = gr.Checkbox(label="Enhance non-face region", value=True)

        with gr.Column():
            gallery = gr.Image(label="Generated Images")
            usage_tips = gr.Markdown(label="Usage tips of InstantID", value=tips ,visible=False)

        submit.click(
            # fn=remove/_tips,
            outputs=usage_tips,            
        ).then(
            # fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            # fn=generate_image,
            inputs=[face_file, pose_file, prompt, negative_prompt, style, num_steps, identitynet_strength_ratio, adapter_strength_ratio, guidance_scale, seed, enable_LCM, enhance_face_region],
            outputs=[gallery, usage_tips]
        )
    
        enable_LCM.input(inputs=[enable_LCM], outputs=[num_steps, guidance_scale], queue=False)

    gr.Examples(
        examples=get_example(),
        inputs=[face_file, prompt, style, negative_prompt],
        run_on_click=True,
        fn=run_for_examples,
        outputs=[gallery, usage_tips],
        cache_examples=True,
    )
    
    gr.Markdown(article)

    demo.launch()