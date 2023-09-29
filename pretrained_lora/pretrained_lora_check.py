import os
from safetensors.torch import load_file, safe_open

pretrained_lora_dir = 'PlatformGameV0_2.safetensors'
weights_sd = load_file(pretrained_lora_dir)
for layer in weights_sd.keys():

    if "alpha" in layer :
        alpha_value = weights_sd[layer]
    elif "lora_down" in layer :
        layer_name = layer.split('.lora')[0]
        down_weight = weights_sd[layer]
        up_layer_key = f'{layer_name}.lora_up.weight'
        up_weight = weights_sd[up_layer_key]
        print(f'down_weight : {down_weight.shape} | up_weight : {up_weight.shape}')