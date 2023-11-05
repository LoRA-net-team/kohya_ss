import torch

dict1 = {'a':1, 'b':2}
dict2 = {'c':3, 'd':4}
dict3 = dict1.update(dict2)
print(dict3)
a = [torch.randn((1,2,3))]
b = torch.randn((1,2,3))
print(torch.cat(a))