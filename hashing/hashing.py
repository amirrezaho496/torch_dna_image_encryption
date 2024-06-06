import hashlib
import torch
import torch.nn as nn
import numpy as np

from randomness.torch_rand import chaotic_torch_rand
from torch_enc.encoding import encode_image_into_4_subcells, encoded_image_into_dna_sequence


MD5= 'MD5'
SHA1 = 'SHA-1'
SHA256 = 'SHA-256'
SHA384 = 'SHA-384'
SHA512 = 'SHA-512'

def hash_function(inp, meth : str):
    # Convert input to bytes
    if isinstance(inp, str):
        inp = inp.encode('utf-8')
    elif isinstance(inp, torch.Tensor):
        inp = bytes(inp.cpu().numpy())
    else:
        inp = bytes(inp)


    # Verify hash method
    meth = meth.upper()

    if meth not in ['MD5', 'SHA-1', 'SHA-256', 'SHA-384', 'SHA-512']:
        raise ValueError("Hash algorithm must be MD5, SHA-1, SHA-256, SHA-384, or SHA-512")


    # Compute hash using the specified hash function
    if meth == MD5:
        h = hashlib.md5(inp).digest()

    elif meth == SHA1:
        h = hashlib.sha1(inp).digest()

    elif meth == SHA256:
        h = hashlib.sha256(inp).digest()

    elif meth == SHA384:
        h = hashlib.sha384(inp).digest()

    elif meth == SHA512:
        h = hashlib.sha512(inp).digest()


    # Convert hash to hexadecimal notation
    h = ''.join(f'{x:02x}' for x in h)

    return h



def hash_image(plain_img : torch.Tensor, key_str: str, meth = SHA256):

    d = "".join([hash_function(plain_img, meth=meth), 
                   hash_function(key_str, meth=meth)])


    h = hash_function(d, meth=meth)
    return h

def create_hash_key(hex1 : str|torch.Tensor , hex2 : str|torch.Tensor, meth = SHA256):
    hash_key = ''.join([hash_function(hex1, meth=meth), 
                            hash_image(hex2 , key_str=meth)])
    
    return ''.join([hash_function(hash_key, meth=meth),
                hash_function(hex2,meth=meth),
                hash_function(hex1, meth=meth)])
    


def hex2bin(h: str, N:int =None, device='cuda:0'):
    """
    Convert hexadecimal string to a binary string.

    Args:
        h (str or list of str): Hexadecimal string(s) to convert.
        N (int, optional): Minimum number of bits in the binary representation.
        device (str, optional): Device to perform the computation on (default: 'cuda:0').

    Returns:
        s (torch.Tensor): Binary representation as a tensor of shape (len(h), N).
    """
    # Ensure h is a tensor on the specified device
    # h = torch.tensor([c for c in h] if isinstance(h, str) else h, device=device)

    # Convert hexadecimal to decimal
    decimal = torch.zeros(len(h), dtype=torch.int)
    for i, c in enumerate(h):
        if c.isdigit():
            decimal[i] = int(c)
        else:
            decimal[i] = ord(c.upper()) - 55

    # Convert decimal to binary
    if N is None:
        N = torch.ceil(torch.log2(decimal.max())).int().item()
    else:
        N = int(N)

    binary = torch.zeros((len(h), N), dtype=torch.int, device=device)
    for i in range(N):
        binary[:, i] = (decimal >> (N - i - 1)) & 1

    return binary

def bin2dec(bin_tensor: torch.Tensor, device='cuda:0'):
    bin_tensor = bin_tensor.to(device)
    return torch.sum(bin_tensor * (2 ** torch.arange(bin_tensor.shape[0] - 1, -1, -1, device=device)))



def hex_str_to_decimal_tensor(hex_str : str, decimal_len : int = 32,  device = 'cuda:0') -> torch.Tensor:
    # Convert each hex digit to an integer
    tensor_id = torch.tensor([int(c, 16) for c in hex_str], device=device, dtype= torch.uint8)
    each_len = tensor_id.size(dim=0) // decimal_len
    
    target = tensor_id.reshape(each_len, -1)
    target = target.sum(dim=0)
    return target

def create_key_image(m,n,key_decimal : torch.Tensor, device = 'cuda:0'):
    torch.manual_seed(key_decimal.sum()+m+n)
    # torch.rand(key_decimal[:16].sum())

    # rands = torch.rand(n*m,dtype=torch.float16).to(device) * 4
    rands = chaotic_torch_rand((n*m,), seed=key_decimal.sum()+m+n, device=device)
    var = torch.floor(rands)
    
    vars4 = encode_image_into_4_subcells(m,n, var, device=device)
    key = encoded_image_into_dna_sequence(m,n, vars4, key_decimal, 100, device=device)
    
    return key