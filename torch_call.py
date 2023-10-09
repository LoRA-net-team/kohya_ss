import os
import torch

torch_dir = r'./test/jungwoo_3_blur_mask_only_down_blocks_0_attentions_1_countinue/jw/heatmap_torch/attention_down_blocks_0_attentions_0'
loaded_torch = torch.load(torch_dir)
print(loaded_torch)