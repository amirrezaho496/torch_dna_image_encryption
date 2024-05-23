import torch

torch.manual_seed(24)
_,perm = torch.rand(10).sort()
tens = torch.tensor(range(0,100)).reshape(10,10)
tens2 = torch.zeros_like(tens)
tens2[:,perm] = tens[:,perm]

print(tens2)