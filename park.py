import os, torch
from torch.nn.functional import normalize

text_encoder_conds = torch.randn((2,77,768))
dim = text_encoder_conds.shape[-1]
caption_attention_mask = torch.zeros((1,77))
caption_attention_mask = [caption_attention_mask,caption_attention_mask]
caption_attention_mask = torch.cat(caption_attention_mask, dim=0).unsqueeze(-1)
caption_attention_mask = torch.repeat_interleave(caption_attention_mask, dim, dim=-1)
print(text_encoder_conds.shape)
print(caption_attention_mask.shape)