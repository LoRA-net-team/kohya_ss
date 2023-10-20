import os
import torch

trg_file = 'up_blocks_1_attentions_0_transformer_blocks_0_attn2_what_map.pt'
what_map_torch = torch.load(trg_file)
uncon, con = what_map_torch.chunk(2, dim=0)
print(f'what_map_torch : {what_map_torch.shape}')
print(f'uncon : {uncon.shape}')