import sys
sys.path.append('./')

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
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers import LCMScheduler

from huggingface_hub import hf_hub_download

import insightface
from insightface.app import FaceAnalysis

from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from pipeline_stable_diffusion_xl_instantid_full import StableDiffusionXLInstantIDPipeline
from gradio_demo.model_util import load_models_xl, get_torch_device, torch_gc

import gradio as gr

import torch
import numpy as np
from PIL import Image
from controlnet_aux import OpenposeDetector
from gradio_demo.model_util import get_torch_device
import cv2


from transformers import DPTImageProcessor, DPTForDepthEstimation

# prepare 'antelopev2' under ./models
app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# prepare models under ./checkpoints
face_adapter = f'./checkpoint/ip-adapter.bin'
controlnet_path = f'./checkpoint/ControlNetModel'

# load IdentityNet
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

base_model = 'wangqixun/YamerMIX_v8'  # from https://civitai.com/models/84040?modelVersionId=196039
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe.cuda()

# load adapter
pipe.load_ip_adapter_instantid(face_adapter)

pipe.enable_model_cpu_offload()

device = get_torch_device()
depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)
feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

from diffusers import LCMScheduler

lcm_lora_path = "./checkpoints/pytorch_lora_weights.safetensors"

pipe.load_lora_weights(lcm_lora_path)
pipe.fuse_lora()
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

num_inference_steps = 10
guidance_scale = 0


style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Watercolor",
        "prompt": "watercolor painting, {prompt}. vibrant, beautiful, painterly, detailed, textural, artistic",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy",
    },
    {
        "name": "Film Noir",
        "prompt": "film noir style, ink sketch|vector, {prompt} highly detailed, sharp focus, ultra sharpness, monochrome, high contrast, dramatic shadows, 1940s style, mysterious, cinematic",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
    },
    {
        "name": "Neon",
        "prompt": "masterpiece painting, buildings in the backdrop, kaleidoscope, lilac orange blue cream fuchsia bright vivid gradient colors, the scene is cinematic, {prompt}, emotional realism, double exposure, watercolor ink pencil, graded wash, color layering, magic realism, figurative painting, intricate motifs, organic tracery, polished",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
    },
    {
        "name": "Jungle",
        "prompt": 'waist-up "{prompt} in a Jungle" by Syd Mead, tangerine cold color palette, muted colors, detailed, 8k,photo r3al,dripping paint,3d toon style,3d style,Movie Still',
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
    },
    {
        "name": "Mars",
        "prompt": "{prompt}, Post-apocalyptic. Mars Colony, Scavengers roam the wastelands searching for valuable resources, rovers, bright morning sunlight shining, (detailed) (intricate) (8k) (HDR) (cinematic lighting) (sharp focus)",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
    },
    {
        "name": "Vibrant Color",
        "prompt": "vibrant colorful, ink sketch|vector|2d colors, at nightfall, sharp focus, {prompt}, highly detailed, sharp focus, the clouds,colorful,ultra sharpness",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
    },
    {
        "name": "Snow",
        "prompt": "cinema 4d render, {prompt}, high contrast, vibrant and saturated, sico style, surrounded by magical glow,floating ice shards, snow crystals, cold, windy background, frozen natural landscape in background  cinematic atmosphere,highly detailed, sharp focus, intricate design, 3d, unreal engine, octane render, CG best quality, highres, photorealistic, dramatic lighting, artstation, concept art, cinematic, epic Steven Spielberg movie still, sharp focus, smoke, sparks, art by pascal blanche and greg rutkowski and repin, trending on artstation, hyperrealism painting, matte painting, 4k resolution",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
    },
    {
        "name": "Line art",
        "prompt": "line art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, vector graphics",
        "negative_prompt": "anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic",
    },
{
    "name": "base",
    "prompt": "{prompt}",
    "negative_prompt": ""
  },
  {
    "name": "3D Model",
    "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
    "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting"
  },
  {
    "name": "Analog Film",
    "prompt": "analog film photo {prompt} . faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage",
    "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
  },
  {
    "name": "Anime",
    "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
    "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast"
  },
  {
    "name": "Cinematic",
    "prompt": "cinematic film still {prompt} . shallow depth of field, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
    "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"
  },
  {
    "name": "Comic Book",
    "prompt": "comic {prompt} . graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
    "negative_prompt": "photograph, deformed, glitch, noisy, realistic, stock photo"
  },
  {
    "name": "Craft Clay",
    "prompt": "play-doh style {prompt} . sculpture, clay art, centered composition, Claymation",
    "negative_prompt": "sloppy, messy, grainy, highly detailed, ultra textured, photo"
  },
  {
    "name": "Digital Art",
    "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
    "negative_prompt": "photo, photorealistic, realism, ugly"
  },
  {
    "name": "Enhance",
    "prompt": "breathtaking {prompt} . award-winning, professional, highly detailed",
    "negative_prompt": "ugly, deformed, noisy, blurry, distorted, grainy"
  },
  {
    "name": "Fantasy Art",
    "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
    "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white"
  },
  {
    "name": "Isometric Style",
    "prompt": "isometric style {prompt} . vibrant, beautiful, crisp, detailed, ultra detailed, intricate",
    "negative_prompt": "deformed, mutated, ugly, disfigured, blur, blurry, noise, noisy, realistic, photographic"
  },
  {
    "name": "Line Art",
    "prompt": "line art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, vector graphics",
    "negative_prompt": "anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic"
  },
  {
    "name": "Lowpoly",
    "prompt": "low-poly style {prompt} . low-poly game art, polygon mesh, jagged, blocky, wireframe edges, centered composition",
    "negative_prompt": "noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo"
  },
  {
    "name": "Neon Punk",
    "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
    "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
  },
  {
    "name": "Origami",
    "prompt": "origami style {prompt} . paper art, pleated paper, folded, origami art, pleats, cut and fold, centered composition",
    "negative_prompt": "noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo"
  },
  {
    "name": "Photographic",
    "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
    "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly"
  },
  {
    "name": "Pixel Art",
    "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
    "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic"
  },
  {
    "name": "Texture",
    "prompt": "texture {prompt} top down close-up",
    "negative_prompt": "ugly, deformed, noisy, blurry"
  },
  {
    "name": "Advertising",
    "prompt": "Advertising poster style {prompt} . Professional, modern, product-focused, commercial, eye-catching, highly detailed",
    "negative_prompt": "noisy, blurry, amateurish, sloppy, unattractive"
  },
  {
    "name": "Food Photography",
    "prompt": "Food photography style {prompt} . Appetizing, professional, culinary, high-resolution, commercial, highly detailed",
    "negative_prompt": "unappetizing, sloppy, unprofessional, noisy, blurry"
  },
  {
    "name": "Real Estate",
    "prompt": "Real estate photography style {prompt} . Professional, inviting, well-lit, high-resolution, property-focused, commercial, highly detailed",
    "negative_prompt": "dark, blurry, unappealing, noisy, unprofessional"
  },
  {
    "name": "Abstract",
    "prompt": "Abstract style {prompt} . Non-representational, colors and shapes, expression of feelings, imaginative, highly detailed",
    "negative_prompt": "realistic, photographic, figurative, concrete"
  },
  {
    "name": "Cubist",
    "prompt": "Cubist artwork {prompt} . Geometric shapes, abstract, innovative, revolutionary",
    "negative_prompt": "anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy"
  },
  {
    "name": "Graffiti",
    "prompt": "Graffiti style {prompt} . Street art, vibrant, urban, detailed, tag, mural",
    "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic"
  },
  {
    "name": "Hyperrealism",
    "prompt": "Hyperrealistic art {prompt} . Extremely high-resolution details, photographic, realism pushed to extreme, fine texture, incredibly lifelike",
    "negative_prompt": "simplified, abstract, unrealistic, impressionistic, low resolution"
  },
  {
    "name": "Impressionist",
    "prompt": "Impressionist painting {prompt} . Loose brushwork, vibrant color, light and shadow play, captures feeling over form",
    "negative_prompt": "anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy"
  },
  {
    "name": "Pointillism",
    "prompt": "Pointillism style {prompt} . Composed entirely of small, distinct dots of color, vibrant, highly detailed",
    "negative_prompt": "line drawing, smooth shading, large color fields, simplistic"
  },
  {
    "name": "Pop Art",
    "prompt": "Pop Art style {prompt} . Bright colors, bold outlines, popular culture themes, ironic or kitsch",
    "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, minimalist"
  },
  {
    "name": "Psychedelic",
    "prompt": "Psychedelic style {prompt} . Vibrant colors, swirling patterns, abstract forms, surreal, trippy",
    "negative_prompt": "monochrome, black and white, low contrast, realistic, photorealistic, plain, simple"
  },
  {
    "name": "Renaissance",
    "prompt": "Renaissance style {prompt} . Realistic, perspective, light and shadow, religious or mythological themes, highly detailed",
    "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, modernist, minimalist, abstract"
  },
  {
    "name": "Steampunk",
    "prompt": "Steampunk style {prompt} . Antique, mechanical, brass and copper tones, gears, intricate, detailed",
    "negative_prompt": "deformed, glitch, noisy, low contrast, anime, photorealistic"
  },
  {
    "name": "Surrealist",
    "prompt": "Surrealist art {prompt} . Dreamlike, mysterious, provocative, symbolic, intricate, detailed",
    "negative_prompt": "anime, photorealistic, realistic, deformed, glitch, noisy, low contrast"
  },
  {
    "name": "Typography",
    "prompt": "Typographic art {prompt} . Stylized, intricate, detailed, artistic, text-based",
    "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic"
  },
  {
    "name": "Watercolor",
    "prompt": "Watercolor painting {prompt} . Vibrant, beautiful, painterly, detailed, textural, artistic",
    "negative_prompt": "anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy"
  },
  {
    "name": "Fighting Game",
    "prompt": "Fighting game style {prompt} . Dynamic, vibrant, action-packed, detailed character design, reminiscent of fighting video games",
    "negative_prompt": "peaceful, calm, minimalist, photorealistic"
  },
  {
    "name": "GTA",
    "prompt": "GTA-style artwork {prompt} . Satirical, exaggerated, pop art style, vibrant colors, iconic characters, action-packed",
    "negative_prompt": "realistic, black and white, low contrast, impressionist, cubist, noisy, blurry, deformed"
  },
  {
    "name": "Super Mario",
    "prompt": "Super Mario style {prompt} . Vibrant, cute, cartoony, fantasy, playful, reminiscent of Super Mario series",
    "negative_prompt": "realistic, modern, horror, dystopian, violent"
  },
  {
    "name": "Minecraft",
    "prompt": "Minecraft style {prompt} . Blocky, pixelated, vibrant colors, recognizable characters and objects, game assets",
    "negative_prompt": "smooth, realistic, detailed, photorealistic, noise, blurry, deformed"
  },
  {
    "name": "Pokémon",
    "prompt": "Pokémon style {prompt} . Vibrant, cute, anime, fantasy, reminiscent of Pokémon series",
    "negative_prompt": "realistic, modern, horror, dystopian, violent"
  },
  {
    "name": "Retro Arcade",
    "prompt": "Retro arcade style {prompt} . 8-bit, pixelated, vibrant, classic video game, old school gaming, reminiscent of 80s and 90s arcade games",
    "negative_prompt": "modern, ultra-high resolution, photorealistic, 3D"
  },
  {
    "name": "Retro Game",
    "prompt": "Retro game art {prompt} . 16-bit, vibrant colors, pixelated, nostalgic, charming, fun",
    "negative_prompt": "realistic, photorealistic, 35mm film, deformed, glitch, low contrast, noisy"
  },
  {
    "name": "RPG Fantasy Game",
    "prompt": "Role-playing game (RPG) style fantasy {prompt} . Detailed, vibrant, immersive, reminiscent of high fantasy RPG games",
    "negative_prompt": "sci-fi, modern, urban, futuristic, low detailed"
  },
  {
    "name": "Strategy Game",
    "prompt": "Strategy game style {prompt} . Overhead view, detailed map, units, reminiscent of real-time strategy video games",
    "negative_prompt": "first-person view, modern, photorealistic"
  },
  {
    "name": "Street Fighter",
    "prompt": "Street Fighter style {prompt} . Vibrant, dynamic, arcade, 2D fighting game, highly detailed, reminiscent of Street Fighter series",
    "negative_prompt": "3D, realistic, modern, photorealistic, turn-based strategy"
  },
  {
    "name": "Legend of Zelda",
    "prompt": "Legend of Zelda style {prompt} . Vibrant, fantasy, detailed, epic, heroic, reminiscent of The Legend of Zelda series",
    "negative_prompt": "sci-fi, modern, realistic, horror"
  },
  {
    "name": "Architectural",
    "prompt": "Architectural style {prompt} . Clean lines, geometric shapes, minimalist, modern, architectural drawing, highly detailed",
    "negative_prompt": "curved lines, ornate, baroque, abstract, grunge"
  },
  {
    "name": "Disco",
    "prompt": "Disco-themed {prompt} . Vibrant, groovy, retro 70s style, shiny disco balls, neon lights, dance floor, highly detailed",
    "negative_prompt": "minimalist, rustic, monochrome, contemporary, simplistic"
  },
  {
    "name": "Dreamscape",
    "prompt": "Dreamscape {prompt} . Surreal, ethereal, dreamy, mysterious, fantasy, highly detailed",
    "negative_prompt": "realistic, concrete, ordinary, mundane"
  },
  {
    "name": "Dystopian",
    "prompt": "Dystopian style {prompt} . Bleak, post-apocalyptic, somber, dramatic, highly detailed",
    "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, cheerful, optimistic, vibrant, colorful"
  },
  {
    "name": "Fairy Tale",
    "prompt": "Fairy tale {prompt} . Magical, fantastical, enchanting, storybook style, highly detailed",
    "negative_prompt": "realistic, modern, ordinary, mundane"
  },
  {
    "name": "Gothic",
    "prompt": "Gothic style {prompt} . Dark, mysterious, haunting, dramatic, ornate, detailed",
    "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, cheerful, optimistic"
  },
  {
    "name": "Grunge",
    "prompt": "Grunge style {prompt} . Textured, distressed, vintage, edgy, punk rock vibe, dirty, noisy",
    "negative_prompt": "smooth, clean, minimalist, sleek, modern, photorealistic"
  },
  {
    "name": "Horror",
    "prompt": "Horror-themed {prompt} . Eerie, unsettling, dark, spooky, suspenseful, grim, highly detailed",
    "negative_prompt": "cheerful, bright, vibrant, light-hearted, cute"
  },
  {
    "name": "Minimalist",
    "prompt": "Minimalist style {prompt} . Simple, clean, uncluttered, modern, elegant",
    "negative_prompt": "ornate, complicated, highly detailed, cluttered, disordered, messy, noisy"
  },
  {
    "name": "Monochrome",
    "prompt": "Monochrome {prompt} . Black and white, contrast, tone, texture, detailed",
    "negative_prompt": "colorful, vibrant, noisy, blurry, deformed"
  },
  {
    "name": "Nautical",
    "prompt": "Nautical-themed {prompt} . Sea, ocean, ships, maritime, beach, marine life, highly detailed",
    "negative_prompt": "landlocked, desert, mountains, urban, rustic"
  },
  {
    "name": "Space",
    "prompt": "Space-themed {prompt} . Cosmic, celestial, stars, galaxies, nebulas, planets, science fiction, highly detailed",
    "negative_prompt": "earthly, mundane, ground-based, realism"
  },
  {
    "name": "Stained Glass",
    "prompt": "Stained glass style {prompt} . Vibrant, beautiful, translucent, intricate, detailed",
    "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic"
  },
  {
    "name": "Techwear Fashion",
    "prompt": "Techwear fashion {prompt} . Futuristic, cyberpunk, urban, tactical, sleek, dark, highly detailed",
    "negative_prompt": "vintage, rural, colorful, low contrast, realism, sketch, watercolor"
  },
  {
    "name": "Tribal",
    "prompt": "Tribal style {prompt} . Indigenous, ethnic, traditional patterns, bold, natural colors, highly detailed",
    "negative_prompt": "modern, futuristic, minimalist, pastel"
  },
  {
    "name": "Zentangle",
    "prompt": "Zentangle {prompt} . Intricate, abstract, monochrome, patterns, meditative, highly detailed",
    "negative_prompt": "colorful, representative, simplistic, large fields of color"
  },
  {
    "name": "Collage",
    "prompt": "Collage style {prompt} . Mixed media, layered, textural, detailed, artistic",
    "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic"
  },
  {
    "name": "Flat Papercut",
    "prompt": "Flat papercut style {prompt} . Silhouette, clean cuts, paper, sharp edges, minimalist, color block",
    "negative_prompt": "3D, high detail, noise, grainy, blurry, painting, drawing, photo, disfigured"
  },
  {
    "name": "Kirigami",
    "prompt": "Kirigami representation of {prompt} . 3D, paper folding, paper cutting, Japanese, intricate, symmetrical, precision, clean lines",
    "negative_prompt": "painting, drawing, 2D, noisy, blurry, deformed"
  },
  {
    "name": "Paper Mache",
    "prompt": "Paper mache representation of {prompt} . 3D, sculptural, textured, handmade, vibrant, fun",
    "negative_prompt": "2D, flat, photo, sketch, digital art, deformed, noisy, blurry"
  },
  {
    "name": "Paper Quilling",
    "prompt": "Paper quilling art of {prompt} . Intricate, delicate, curling, rolling, shaping, coiling, loops, 3D, dimensional, ornamental",
    "negative_prompt": "photo, painting, drawing, 2D, flat, deformed, noisy, blurry"
  },
  {
    "name": "Papercut Collage",
    "prompt": "Papercut collage of {prompt} . Mixed media, textured paper, overlapping, asymmetrical, abstract, vibrant",
    "negative_prompt": "photo, 3D, realistic, drawing, painting, high detail, disfigured"
  },
  {
    "name": "Papercut Shadow Box",
    "prompt": "3D papercut shadow box of {prompt} . Layered, dimensional, depth, silhouette, shadow, papercut, handmade, high contrast",
    "negative_prompt": "painting, drawing, photo, 2D, flat, high detail, blurry, noisy, disfigured"
  },
  {
    "name": "Stacked Papercut",
    "prompt": "Stacked papercut art of {prompt} . 3D, layered, dimensional, depth, precision cut, stacked layers, papercut, high contrast",
    "negative_prompt": "2D, flat, noisy, blurry, painting, drawing, photo, deformed"
  },
  {
    "name": "Thick Layered Papercut",
    "prompt": "Thick layered papercut art of {prompt} . Deep 3D, volumetric, dimensional, depth, thick paper, high stack, heavy texture, tangible layers",
    "negative_prompt": "2D, flat, thin paper, low stack, smooth texture, painting, drawing, photo, deformed"
  },
  {
    "name": "Alien",
    "prompt": "Alien-themed {prompt} . Extraterrestrial, cosmic, otherworldly, mysterious, sci-fi, highly detailed",
    "negative_prompt": "earthly, mundane, common, realistic, simple"
  },
  {
    "name": "Film Noir",
    "prompt": "Film noir style {prompt} . Monochrome, high contrast, dramatic shadows, 1940s style, mysterious, cinematic",
    "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, vibrant, colorful"
  },
  {
    "name": "HDR",
    "prompt": "HDR photo of {prompt} . High dynamic range, vivid, rich details, clear shadows and highlights, realistic, intense, enhanced contrast, highly detailed",
    "negative_prompt": "flat, low contrast, oversaturated, underexposed, overexposed, blurred, noisy"
  },
  {
    "name": "Long Exposure",
    "prompt": "Long exposure photo of {prompt} . Blurred motion, streaks of light, surreal, dreamy, ghosting effect, highly detailed",
    "negative_prompt": "static, noisy, deformed, shaky, abrupt, flat, low contrast"
  },
  {
    "name": "Neon Noir",
    "prompt": "Neon noir {prompt} . Cyberpunk, dark, rainy streets, neon signs, high contrast, low light, vibrant, highly detailed",
    "negative_prompt": "bright, sunny, daytime, low contrast, black and white, sketch, watercolor"
  },
  {
    "name": "Silhouette",
    "prompt": "Silhouette style {prompt} . High contrast, minimalistic, black and white, stark, dramatic",
    "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, color, realism, photorealistic"
  },
  {
    "name": "Tilt-Shift",
    "prompt": "Tilt-shift photo of {prompt} . Selective focus, miniature effect, blurred background, highly detailed, vibrant, perspective control",
    "negative_prompt": "blurry, noisy, deformed, flat, low contrast, unrealistic, oversaturated, underexposed"
  }
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}

def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

def get_canny_image(image, t1=100, t2=200):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    edges = cv2.Canny(image, t1, t2)
    return Image.fromarray(edges, "L")

# global variable
MAX_SEED = np.iinfo(np.int32).max
device = get_torch_device()
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Watercolor"

# Load face encoder
app = FaceAnalysis(
    name="antelopev2",
    root="./",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(640, 640))

# Path to InstantID models
face_adapter = f"./checkpoints/ip-adapter.bin"
controlnet_path = f"./checkpoints/ControlNetModel"

# Load pipeline face ControlNetModel
controlnet_identitynet = ControlNetModel.from_pretrained(
    controlnet_path, torch_dtype=dtype
)

# controlnet-pose
controlnet_pose_model = "thibaud/controlnet-openpose-sdxl-1.0"
controlnet_canny_model = "diffusers/controlnet-canny-sdxl-1.0"
controlnet_depth_model = "diffusers/controlnet-depth-sdxl-1.0-small"

controlnet_pose = ControlNetModel.from_pretrained(
    controlnet_pose_model, torch_dtype=dtype
).to(device)
controlnet_canny = ControlNetModel.from_pretrained(
    controlnet_canny_model, torch_dtype=dtype
).to(device)
controlnet_depth = ControlNetModel.from_pretrained(
    controlnet_depth_model, torch_dtype=dtype
).to(device)

controlnet_map = {
    "pose": controlnet_pose,
    "canny": controlnet_canny,
    "depth": controlnet_depth,
}
controlnet_map_fn = {
    "pose": openpose,
    "canny": get_canny_image,
    "depth": get_depth_map,
}

pretrained_model_name_or_path = "wangqixun/YamerMIX_v8"
enable_LCM = False
enable_lcm_arg=False

# def main(pretrained_model_name_or_path="wangqixun/YamerMIX_v8", enable_lcm_arg=False):
if pretrained_model_name_or_path.endswith(
    ".ckpt"
) or pretrained_model_name_or_path.endswith(".safetensors"):
    scheduler_kwargs = hf_hub_download(
        repo_id="wangqixun/YamerMIX_v8",
        subfolder="scheduler",
        filename="scheduler_config.json",
    )

    (tokenizers, text_encoders, unet, _, vae) = load_models_xl(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        scheduler_name=None,
        weight_dtype=dtype,
    )

    scheduler = diffusers.EulerDiscreteScheduler.from_config(scheduler_kwargs)
    pipe = StableDiffusionXLInstantIDPipeline(
        vae=vae,
        text_encoder=text_encoders[0],
        text_encoder_2=text_encoders[1],
        tokenizer=tokenizers[0],
        tokenizer_2=tokenizers[1],
        unet=unet,
        scheduler=scheduler,
        controlnet=[controlnet_identitynet],
    ).to(device)

else:
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        pretrained_model_name_or_path,
        controlnet=[controlnet_identitynet],
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
    ).to(device)

    pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(
        pipe.scheduler.config
    )

pipe.load_ip_adapter_instantid(face_adapter)
# load and disable LCM
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
pipe.disable_lora()

def toggle_lcm_ui(value):
    if value:
        return (
            gr.update(minimum=0, maximum=100, step=1, value=5),
            gr.update(minimum=0.1, maximum=20.0, step=0.1, value=1.5),
        )
    else:
        return (
            gr.update(minimum=5, maximum=100, step=1, value=30),
            gr.update(minimum=0.1, maximum=20.0, step=0.1, value=5),
        )

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def remove_tips():
    return gr.update(visible=False)

def get_example():
    case = [
        [
            "./examples/yann.jpg",
            "./examples/poses/pose1.jpg",
            "a man",
            "Neon",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
        [
            "./examples/Tom.png",
            "./examples/poses/pose2.jpg",
            "a man flying in the sky in Mars",
            "Jungle",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
        [
            "./examples/Scott.png",
            "./examples/poses/pose3.jpg",
            "a man doing a silly pose wearing a suite",
            "Neon Noir",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, gree",
        ],
        [
            "./examples/Paul.jpeg",
            "./examples/poses/pose4.jpg",
            "a man sit on a chair",
            "Vibrant Color",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
        [
            "./examples/kevin.png",
            "./examples/poses/pose5.png",
            "a man doing a silly pose wearing a suite",
            "GTA",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
        [
            "./examples/001.png",
            "./examples/poses/pose6.png",
            "a man doing a silly pose wearing a suite",
            "Street Fighter",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, gree",
        ],
        [
            "./examples/002.png",
            "./examples/poses/pose7.jpg",
            "a woman sit on a chair",
            "Disco",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
        [
            "./examples/003.png",
            "./examples/poses/pose8.png",
            "a man",
            "Dreamscape",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
    ]
    return case

def run_for_examples(face_file, pose_file, prompt, style, negative_prompt):
    return generate_image(
        face_file,
        pose_file,
        prompt,
        negative_prompt,
        style,
        20,  # num_steps
        0.8,  # identitynet_strength_ratio
        0.8,  # adapter_strength_ratio
        0.4,  # pose_strength
        0.3,  # canny_strength
        0.5,  # depth_strength
        ["pose", "canny"],  # controlnet_selection
        5.0,  # guidance_scale
        42,  # seed
        "EulerDiscreteScheduler",  # scheduler
        False,  # enable_LCM
        True,  # enable_Face_Region
    )

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def draw_kps(
    image_pil,
    kps,
    color_list=[
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ],
):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))),
            (int(length / 2), stickwidth),
            int(angle),
            0,
            360,
            1,
        )
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

def resize_img(
    input_image,
    max_side=1280,
    min_side=1024,
    size=None,
    pad_to_max_side=False,
    mode=PIL.Image.BILINEAR,
    base_pixel_number=64,
):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[
            offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new
        ] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

def apply_style(
    style_name: str, positive: str, negative: str = ""
) -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + " " + negative

def generate_image(
    face_image_path,
    pose_image_path,
    prompt,
    negative_prompt,
    style_name,
    num_steps,
    identitynet_strength_ratio,
    adapter_strength_ratio,
    pose_strength,
    canny_strength,
    depth_strength,
    controlnet_selection,
    guidance_scale,
    seed,
    scheduler,
    enable_LCM,
    enhance_face_region,
    progress=gr.Progress(track_tqdm=True),
):

    if enable_LCM:
        pipe.scheduler = diffusers.LCMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_lora()
    else:
        pipe.disable_lora()
        scheduler_class_name = scheduler.split("-")[0]

        add_kwargs = {}
        if len(scheduler.split("-")) > 1:
            add_kwargs["use_karras_sigmas"] = True
        if len(scheduler.split("-")) > 2:
            add_kwargs["algorithm_type"] = "sde-dpmsolver++"
        scheduler = getattr(diffusers, scheduler_class_name)
        pipe.scheduler = scheduler.from_config(pipe.scheduler.config, **add_kwargs)

    if face_image_path is None:
        raise gr.Error(
            f"Cannot find any input face image! Please upload the face image"
        )

    if prompt is None:
        prompt = "a person"

    # apply the style template
    prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

    face_image = load_image(face_image_path)
    face_image = resize_img(face_image, max_side=1024)
    face_image_cv2 = convert_from_image_to_cv2(face_image)
    height, width, _ = face_image_cv2.shape

    # Extract face features
    face_info = app.get(face_image_cv2)

    if len(face_info) == 0:
        raise gr.Error(
            f"Unable to detect a face in the image. Please upload a different photo with a clear face."
        )

    face_info = sorted(
        face_info,
        key=lambda x: (x["bbox"][2] - x["bbox"][0]) * x["bbox"][3] - x["bbox"][1],
    )[
        -1
    ]  # only use the maximum face
    face_emb = face_info["embedding"]
    face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info["kps"])
    img_controlnet = face_image
    if pose_image_path is not None:
        pose_image = load_image(pose_image_path)
        pose_image = resize_img(pose_image, max_side=1024)
        img_controlnet = pose_image
        pose_image_cv2 = convert_from_image_to_cv2(pose_image)

        face_info = app.get(pose_image_cv2)

        if len(face_info) == 0:
            raise gr.Error(
                f"Cannot find any face in the reference image! Please upload another person image"
            )

        face_info = face_info[-1]
        face_kps = draw_kps(pose_image, face_info["kps"])

        width, height = face_kps.size

    if enhance_face_region:
        control_mask = np.zeros([height, width, 3])
        x1, y1, x2, y2 = face_info["bbox"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        control_mask[y1:y2, x1:x2] = 255
        control_mask = Image.fromarray(control_mask.astype(np.uint8))
    else:
        control_mask = None

    if len(controlnet_selection) > 0:
        controlnet_scales = {
            "pose": pose_strength,
            "canny": canny_strength,
            "depth": depth_strength,
        }
        pipe.controlnet = MultiControlNetModel(
            [controlnet_identitynet]
            + [controlnet_map[s] for s in controlnet_selection]
        )
        control_scales = [float(identitynet_strength_ratio)] + [
            controlnet_scales[s] for s in controlnet_selection
        ]
        control_images = [face_kps] + [
            controlnet_map_fn[s](img_controlnet).resize((width, height))
            for s in controlnet_selection
        ]
    else:
        pipe.controlnet = controlnet_identitynet
        control_scales = float(identitynet_strength_ratio)
        control_images = face_kps

    generator = torch.Generator(device=device).manual_seed(seed)

    print("Start inference...")
    print(f"[Debug] Prompt: {prompt}, \n[Debug] Neg Prompt: {negative_prompt}")

    pipe.set_ip_adapter_scale(adapter_strength_ratio)
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_embeds=face_emb,
        image=control_images,
        control_mask=control_mask,
        controlnet_conditioning_scale=control_scales,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    ).images

    return images[0], gr.update(visible=True)

# Description
title = r"""
<h1 align="center">InstantID: Zero-shot Identity-Preserving Generation in Seconds</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://github.com/InstantID/InstantID' target='_blank'><b>InstantID: Zero-shot Identity-Preserving Generation in Seconds</b></a>.<br>

How to use:<br>
1. Upload an image with a face. For images with multiple faces, we will only detect the largest face. Ensure the face is not too small and is clearly visible without significant obstructions or blurring.
2. (Optional) You can upload another image as a reference for the face pose. If you don't, we will use the first detected face image to extract facial landmarks. If you use a cropped face at step 1, it is recommended to upload it to define a new face pose.
3. (Optional) You can select multiple ControlNet models to control the generation process. The default is to use the IdentityNet only. The ControlNet models include pose skeleton, canny, and depth. You can adjust the strength of each ControlNet model to control the generation process.
4. Enter a text prompt, as done in normal text-to-image models.
5. Click the <b>Submit</b> button to begin customization.
6. Share your customized photo with your friends and enjoy! 😊"""

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

css = """
.gradio-container {width: 85% !important}
"""
with gr.Blocks(css=css) as demo:
    # description
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            with gr.Row(equal_height=True):
                # upload face image
                face_file = gr.Image(
                    label="Upload a photo of your face", type="filepath"
                )
                # optional: upload a reference pose image
                pose_file = gr.Image(
                    label="Upload a reference pose image (Optional)",
                    type="filepath",
                )

            # prompt
            prompt = gr.Textbox(
                label="Prompt",
                info="Give simple prompt is enough to achieve good face fidelity",
                placeholder="A photo of a person",
                value="",
            )

            submit = gr.Button("Submit", variant="primary")
            enable_LCM = gr.Checkbox(
                label="Enable Fast Inference with LCM", value=enable_lcm_arg,
                info="LCM speeds up the inference step, the trade-off is the quality of the generated image. It performs better with portrait face images rather than distant faces",
            )
            style = gr.Dropdown(
                label="Style template",
                choices=STYLE_NAMES,
                value=DEFAULT_STYLE_NAME,
            )

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
            with gr.Accordion("Controlnet"):
                controlnet_selection = gr.CheckboxGroup(
                    ["pose", "canny", "depth"], label="Controlnet", value=["pose"],
                    info="Use pose for skeleton inference, canny for edge detection, and depth for depth map estimation. You can try all three to control the generation process"
                )
                pose_strength = gr.Slider(
                    label="Pose strength",
                    minimum=0,
                    maximum=1.5,
                    step=0.05,
                    value=0.40,
                )
                canny_strength = gr.Slider(
                    label="Canny strength",
                    minimum=0,
                    maximum=1.5,
                    step=0.05,
                    value=0.40,
                )
                depth_strength = gr.Slider(
                    label="Depth strength",
                    minimum=0,
                    maximum=1.5,
                    step=0.05,
                    value=0.40,
                )
            with gr.Accordion(open=False, label="Advanced Options"):
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="low quality",
                    value="(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
                )
                num_steps = gr.Slider(
                    label="Number of sample steps",
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=5 if enable_lcm_arg else 30,
                )
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.1,
                    maximum=20.0,
                    step=0.1,
                    value=0.0 if enable_lcm_arg else 5.0,
                )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=42,
                )
                schedulers = [
                    "DEISMultistepScheduler",
                    "HeunDiscreteScheduler",
                    "EulerDiscreteScheduler",
                    "DPMSolverMultistepScheduler",
                    "DPMSolverMultistepScheduler-Karras",
                    "DPMSolverMultistepScheduler-Karras-SDE",
                ]
                scheduler = gr.Dropdown(
                    label="Schedulers",
                    choices=schedulers,
                    value="EulerDiscreteScheduler",
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                enhance_face_region = gr.Checkbox(label="Enhance non-face region", value=True)

        with gr.Column(scale=1):
            gallery = gr.Image(label="Generated Images")
            usage_tips = gr.Markdown(
                label="InstantID Usage Tips", value=tips, visible=False
            )

        submit.click(
            fn=remove_tips,
            outputs=usage_tips,
        ).then(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=generate_image,
            inputs=[
                face_file,
                pose_file,
                prompt,
                negative_prompt,
                style,
                num_steps,
                identitynet_strength_ratio,
                adapter_strength_ratio,
                pose_strength,
                canny_strength,
                depth_strength,
                controlnet_selection,
                guidance_scale,
                seed,
                scheduler,
                enable_LCM,
                enhance_face_region,
            ],
            outputs=[gallery, usage_tips],
        )

        enable_LCM.input(
            fn=toggle_lcm_ui,
            inputs=[enable_LCM],
            outputs=[num_steps, guidance_scale],
            queue=False,
        )

    gr.Examples(
        examples=get_example(),
        inputs=[face_file, pose_file, prompt, style, negative_prompt],
        fn=run_for_examples,
        outputs=[gallery, usage_tips],
        cache_examples=True,
    )

    gr.Markdown(article)

demo.launch(share=True)



# main(pretrained_model_name_or_path, enable_LCM)