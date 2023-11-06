import os, torch
from torch.nn.functional import normalize

text_embeddings = torch.randn((1,77,768))
previous_mean = text_embeddings.float().mean(axis=[-2, -1])
#print(previous_mean)
previous_mean = text_embeddings.float().mean()
#print(previous_mean)

prompt_weights = torch.ones((1,77,1))
prompt_weights = prompt_weights.squeeze(-1)
print(prompt_weights.shape)