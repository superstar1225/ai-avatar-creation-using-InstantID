from huggingface_hub import hf_hub_download
import gdown
import os
import zipfile

# download models
hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/config.json",
    local_dir="./checkpoints",
)
hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/diffusion_pytorch_model.safetensors",
    local_dir="./checkpoints",
)
hf_hub_download(
    repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="./checkpoints"
)
hf_hub_download(
    repo_id="latent-consistency/lcm-lora-sdxl",
    filename="pytorch_lora_weights.safetensors",
    local_dir="./checkpoints",
)
# download antelopev2
gdown.download(url="https://drive.google.com/uc?id=18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8", output="./models/", quiet=False, fuzzy=True)
# unzip antelopev2.zip
# os.system("unzip ./model/antelopev2.zip -d ./model/")
with zipfile.ZipFile('./models/antelopev2.zip', mode='r') as z:
    z.extractall('models')