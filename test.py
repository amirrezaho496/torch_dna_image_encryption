import torch

from randomness.torch_rand import chaotic_torch_rand

# torch.manual_seed(24)
# _,perm = torch.rand(10).sort()

# torch.manual_seed(24)
# _,perm2 = torch.rand(10).to(device='cuda').sort()
# # tens = torch.tensor(range(0,100)).reshape(10,10)
# # tens2 = torch.zeros_like(tens)
# # tens2[:,perm] = tens[:,perm]

rand = chaotic_torch_rand(size=(25,), seed=10, device='cuda:0') 


print(rand)


# tensor([0.7713, 0.0208, 0.6336, 0.7488, 0.4985, 0.7034, 0.0810, 0.9257, 0.7501,
#         0.9969, 0.8320, 0.2970, 0.2742, 0.7475, 0.0121, 0.5574, 0.8326, 0.7936,
#         0.7526, 0.0479, 0.9838, 0.5559, 0.6531, 0.7425, 0.1817],
#        device='cuda:0')