import torch
from torch.nn.functional import normalize



effective_text_embedding = torch.tensor([[-1.2,2.1,3.0],
                                          [3.0,4.0,5.0]])
a = normalize(effective_text_embedding, p=2,dim=1)
print(a)