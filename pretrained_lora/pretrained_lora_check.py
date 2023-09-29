import os
from safetensors.torch import load_file, safe_open

pretrained_lora_dir = 'PlatformGameV0_2.safetensors'
weights_sd = load_file(pretrained_lora_dir)
for layer in weights_sd.keys():
    print(layer)