import builtins
from typing import Sequence, Union
import numpy as np
import torch

_int = builtins.int

from random import seed
from random import random

def chaotic_torch_rand(size : Sequence[_int|torch.SymInt], seed = 0, r : torch.uint8 = 200, device : str|None = None):

    if device == None:
        device = 'cpu'
    
    count = 1
    for dim in size:
        count *= dim

    # col = int(np.ceil(np.sqrt(count)))
    dim1 = int(np.ceil(np.log2(count)))
    
    dim2 = int(np.ceil(count / dim1))
    
    torch.manual_seed(seed)
    first_rand= torch.rand(dim2, dtype=torch.float, device='cpu')
    
    x = torch.zeros(size=(dim1,dim2)).to(device=device, dtype=torch.float)
    
    x[0,:] = first_rand
    # _,perm = first_rand.sort()
    # x[0] = torch.zeros(dim2) + 0.1111
    
    r = 3.98 + ( r / (256 * 20))
    for i in range(1,dim1):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
        # x[i, perm] = r * x[i - 1] * (1 - x[i - 1])

    x = x.reshape(-1)[:count]
    return x.reshape(size)