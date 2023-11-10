import os, torch
from torch.nn.functional import normalize

text_encoder_conds = torch.randn((2,77,768))

caption_attention_mask = torch.randn((1,77))
caption_attention_mask = caption_attention_mask.unsqueeze(-1)