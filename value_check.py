import os
import torch, math
from einops import rearrange

trg_file = 'up_blocks_1_attentions_0_transformer_blocks_0_attn2_what_map.pt'
what_map_torch = torch.load(trg_file)
uncon, con = what_map_torch.chunk(2, dim=0)
print(f'what_map_torch : {what_map_torch.shape}')
pix_len = con.shape[1]
res = math.sqrt(pix_len)
re_con = rearrange(con, 'b (r r) c -> b r r w')
print(f're_con : {re_con.shape}')