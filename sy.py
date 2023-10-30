import torch
a = torch.randn((1,2,3))
b = torch.randn((1,2,3))
list_torches_k = [a,b]
d = torch.stack(list_torches_k)
c = torch.mean(torch.stack(list_torches_k), dim=0)

print(a)
print(b)
print(d)
print(c)