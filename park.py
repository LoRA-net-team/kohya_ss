import os, torch
from torch.nn.functional import normalize
org_text_input = torch.tensor([[[1.0,2.0,3.0],
                                [3.0,4.0,5.0]]])
norm_text_input = normalize(org_text_input, p=2, dim=2)
#trg_size = torch.norm(org_text_input, dim=2).unsqueeze(-1)

trg_size = torch.ones(org_text_input.shape)
trg_size[:,1,:] = 0.8
print(trg_size)
#trg_size = trg_size.expand(org_text_input.shape)
print(trg_size)
print(org_text_input)
print(norm_text_input)
print(trg_size)
print()
out = norm_text_input*trg_size
print(out)