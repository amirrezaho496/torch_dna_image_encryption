import numpy as np
import torch
import torch.nn as nn


def encode_image_into_4_subcells(m: int, n: int, plain_img: torch.Tensor, device = 'cuda:0') -> torch.Tensor:
    """
    Encodes each cell of the given image into 4 subcells.

    Args:
        m: The height of the image
        n: The width of the image
        plain_img: The original image to be encoded

    Returns:
        The encoded image
    """
    # Flatten the image
    plain_img = plain_img.reshape(-1)

    # Initialize the output tensor
    I = torch.zeros(4 * n * m, dtype=torch.int8, device=device)
    
    # Encode the image into 4 subcells
    num2decomposed = plain_img
    for z in range(1, 5):
        rem = num2decomposed % 4
        I[(4 - z)::4] = rem
        num2decomposed = num2decomposed // 4

    return I


def encoded_image_into_dna_sequence(m: int, n: int, I: torch.Tensor, KeyDecimal: torch.Tensor, KeyFeature: int, device = 'cuda:0') -> torch.Tensor:
    """
    Encodes the image into a DNA sequence using the given key decimal and feature values.

    Args:
        m: The height of the image
        n: The width of the image
        I: The encoded image
        KeyDecimal: The key decimal values
        KeyFeature: The key feature value

    Returns:
        The encoded DNA sequence
    """
    # Extract the key decimal values
    d1, d2, d3, d4, d5, d6, d7, d8 = KeyDecimal[0], KeyDecimal[1], KeyDecimal[2], KeyDecimal[3], KeyDecimal[4], KeyDecimal[5], KeyDecimal[6], KeyDecimal[7]
    d9, d10, d11, d12, d13, d14, d15, d16 = KeyDecimal[8], KeyDecimal[9], KeyDecimal[10], KeyDecimal[11], KeyDecimal[12], KeyDecimal[13], KeyDecimal[14], KeyDecimal[15]

    # Calculate xx and u
    xx = torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(d1, d2), d3), d4), d5), d6), d7), d8), KeyFeature) / 256
    u = 3.89 + xx * 0.01

    # Calculate x
    x = torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(d9, d10), d11), d12), d13), d14), d15), d16), KeyFeature) / 256

    # Generate the logistic sequence for the entire image
    len4mn = 4 * n * m
    torch.manual_seed(x*100 + u *100)
    torch.cuda.manual_seed(x* 100 + u * 100)
    logistic_seq = torch.rand((1, len4mn)).to(device=device)

    # Calculate R
    R = torch.floor(8 * logistic_seq).int()

    RULES = torch.tensor([
        [1, 0, 3, 2],# r = 1
        [1, 3, 0, 2],# r = 2
        [0, 1, 2, 3],# r = 3
        [0, 2, 1, 3],# r = 4
        [3, 1, 2, 0],# r = 5
        [3, 2, 1, 0],# r = 6
        [2, 0, 3, 1],# r = 7
        [2, 3, 0, 1] # r = 8
    ], device=device, dtype= torch.uint8)
    # Encode the DNA sequence    
    encode_dna = RULES.to(device=device)[R,I.int()]

    return encode_dna


def permutation_dna(image: torch.Tensor, key_decimal: torch.Tensor, key_feature: int, m: int, n: int, type: str, step_m = 0 , step_n= 0, device = 'cuda:0') -> torch.Tensor:
    """
    Permutes the DNA sequence using the given key decimal and feature values.

    Args:
        image: The DNA sequence to be permuted
        key_decimal: The key decimal values
        key_feature: The key feature value
        m: The height of the image
        n: The width of the image
        type: The type of permutation (Encryption or Decryption)

    Returns:
        The permuted DNA sequence
    """

    seed = key_decimal[1::2].sum() * step_m + key_decimal[0::2].sum() * step_n + key_feature
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    nn_module = torch.nn.Sequential(
        nn.Linear(len(key_decimal), 12),
        nn.Tanh(),
        nn.Linear(12,2),
        nn.Sigmoid()
    ).to(device=device)

    x,u = nn_module(key_decimal.float())
    
    len4mn = 4 * n * m
    torch.manual_seed(x * 10 + u * 100 + key_feature)
    torch.cuda.manual_seed(x * 100 +  u * 100 + key_feature)
    chaotic_signal = torch.rand(len4mn,dtype=torch.float16).to(device=device)

    
    # # Permute the image
    _, pos = torch.sort(chaotic_signal)
    _ = None
    pos = pos.int()
    if type == 'encryption':
        per_image = image.squeeze(dim=0)[pos]
    elif type == 'decryption':
        per_image = torch.zeros(len4mn, dtype=torch.uint8, device=device)
        per_image[pos] = image

    return per_image


def diffusion_dna(image: torch.Tensor, key_image: torch.Tensor, key_decimal: torch.Tensor, key_feature: int, m: int, n: int, type: str,step_m = 0, step_n = 0, device = 'cuda:0') -> torch.Tensor:
    """
    Performs diffusion on the DNA sequence using the given key decimal and feature values.

    Args:
        image: The DNA sequence to be diffused
        key_image: The key image used for diffusion
        key_decimal: The key decimal values
        key_feature: The key feature value
        m: The height of the image
        n: The width of the image
        type: The type of diffusion (Encryption or Decryption)

    Returns:
        The diffused DNA sequence
    """
    half = len(key_decimal//2)
    seed = key_decimal[:half].sum() * step_m + key_decimal[half:].sum() * step_n + key_feature
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    nn_module = torch.nn.Sequential(
        nn.Linear(len(key_decimal), 16),
        nn.Tanh(),
        nn.Linear(16,2),
        nn.Sigmoid()
    ).to(device=device, dtype=torch.float)
    
    x, u = nn_module(key_decimal.float())
    
    # Generate the chaotic signal for the entire image
    len4mn = 4 * n * m
    torch.manual_seed(x*step_m + u*step_n )
    torch.cuda.manual_seed(x*step_m + u * step_n)
    chaotic_signal = torch.rand((1, len4mn)).to(device=device)

    # Calculate the operation
    operation = torch.floor(7 * chaotic_signal).int()

    # Perform the diffusion operation
    image = image.int().squeeze()
    key_image = key_image.int()
    
    DNAOPR = torch.tensor([
        [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]],  #0 XOR
        [[1, 0, 3, 2], [0, 1, 2, 3], [3, 2, 1, 0], [2, 3, 0, 1]],  #1 ADD
        [[3, 2, 1, 0], [2, 3, 0, 1], [1, 0, 3, 2], [0, 1, 2, 3]],  #2 MUL
        [[3, 2, 1, 0], [2, 3, 0, 1], [1, 0, 3, 2], [0, 1, 2, 3]],  #3 XNOR
        [[1, 2, 3, 0], [0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1]],  #4 SUB
        [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]],  #5 RShift
        [[0, 3, 2, 1], [1, 0, 3, 2], [2, 1, 0, 3], [3, 2, 1, 0]]   #6 LShift
    ], dtype=torch.uint8, device=device)

    if type == 'encryption':
        op = operation

    elif type == 'decryption':
        op = operation.clone()
        op[operation == 5] = 6
        op[operation == 6] = 5
        
    diff_img = DNAOPR[op, image, key_image]

    return diff_img


def decoding_dna_image(m: int, n: int, I: torch.Tensor, key_decimal: torch.Tensor, key_feature: int, device = 'cuda:0') -> torch.Tensor:

    """

    Decodes a DNA sequence into an image using the given key decimal and feature values.


    Args:

        m: The height of the image

        n: The width of the image

        I: The DNA sequence to be decoded

        key_decimal: The key decimal values

        key_feature: The key feature value


    Returns:

        The decoded image

    """

    # Extract the key decimal values
    d1, d2, d3, d4, d5, d6, d7, d8 = key_decimal[0], key_decimal[1], key_decimal[2], key_decimal[3], key_decimal[4], key_decimal[5], key_decimal[6], key_decimal[7]
    d9, d10, d11, d12, d13, d14, d15, d16 = key_decimal[8], key_decimal[9], key_decimal[10], key_decimal[11], key_decimal[12], key_decimal[13], key_decimal[14], key_decimal[15]


    # Calculate xx and u

    xx = torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(d1, d2), d3), d4), d5), d6), d7), d8), key_feature) / 256
    u = 3.89 + xx * 0.01


    # Calculate x
    x = torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(torch.bitwise_xor(d9, d10), d11), d12), d13), d14), d15), d16), key_feature) / 256



    # Generate the logistic sequence for the entire image
    len4mn = 4 * n * m
    torch.manual_seed(x*100 + u *100)
    torch.cuda.manual_seed(x*100 +u * 100)
    logistic_seq = torch.rand((1, len4mn)).to(device=device)

    # Calculate R
    R = torch.floor(8 * logistic_seq).int()


    # Decode the DNA sequence
    decode_dna = torch.zeros(len4mn, dtype=torch.int8, device= device)

    # Create a tensor for the mapping
    RULES = torch.tensor([
        [1, 0, 3, 2], # r = 1
        [2, 0, 3, 1], # r = 2
        [0, 1, 2, 3], # r = 3 
        [0, 2, 1, 3], # r = 4
        [3, 1, 2, 0], # r = 5
        [3, 2, 1, 0], # r = 6
        [1, 3, 0, 2], # r = 7
        [2, 3, 0, 1]  # r = 8
    ]).to(device=device, dtype=torch.uint8)  # Move to CUDA device

    # Perform the mapping using broadcasting
    decode_dna = RULES.to(device=device)[R, I.int()].squeeze()
    
    # num = torch.zeros(size = (1, m*n), dtype=torch.uint8, device=device).squeeze(dim=0)


    num = decode_dna[0::4]*64 + decode_dna[1::4]*16 + decode_dna[2::4]*4 +  decode_dna[3::4]
  
    # decoded_image = num

    # # Reshape Imagedecoding tensor
    decoded_image = num.view(m,n)
    
    # Return the decoded image
    return decoded_image


def permutation_by_gcd(img:torch.Tensor, key_decimal : torch.Tensor, type : str,  device = 'cuda:0'):
    m,n = img.shape
    gcd = np.gcd(m,n)
    if (gcd < 4):
        return img
    
    for i in range(200, 10, -1):
        if gcd % i == 0:
            gcd /= i
            break
    gcd = int(gcd)
    col_size:int = n // gcd
    row_size:int = m // gcd
    
    seed = key_decimal[0::3].sum() * 100*gcd + key_decimal[1::3].sum() * m + key_decimal[2::3].sum() * n
    seed //= m+n+100*gcd
    torch.manual_seed(seed)
    groupe_size = int(col_size*row_size)
    rands = torch.rand(size=(1,groupe_size)).to(device=device)
    _, perm = torch.sort(rands)
    
    # gcd_rands = torch.rand(size=(2,gcd), device=device)
    # _, gcd_perms = torch.sort(gcd_rands)

    org_img = img.reshape(groupe_size,gcd,gcd)
    
    
    if type == 'encryption':
        per_image = org_img[perm.squeeze()]
        # per_image = per_image.clone()[:,gcd_perms[0]]
        # per_image = per_image.clone()[:,:,gcd_perms[1]]
    elif type == 'decryption':
        per_image = torch.zeros_like(org_img, device=device, dtype=torch.uint8)
        # per_image[:,:,gcd_perms[1]] = org_img
        # per_image[:,gcd_perms[0]] = per_image.clone()
        per_image[perm.squeeze()] = org_img
        
    per_image = per_image.reshape(m,n)
    return per_image


def permutation_rows(img : torch.Tensor, key_decimal : torch.Tensor, type :str,  device = 'cuda:0'):
    m,n = img.shape
    
    seed = key_decimal[0]
    for i in key_decimal:
        seed = seed.bitwise_xor(i).bitwise_not()
        
    torch.manual_seed(seed)
        
    _, perms = torch.rand((1,m)).to(device=device).squeeze().sort()
    
    if type == 'encryption':
         perm_img = img[perms,:]
    elif type == 'decryption' :
         perm_img = torch.zeros_like(img, device=device)
         perm_img[perms,:] = img
    
    return perm_img.to(torch.uint8)

def permutation_columns(img : torch.Tensor, key_decimal : torch.Tensor, type :str,  device = 'cuda:0'):
    m,n = img.shape
    
    seed = key_decimal[0]
    for i in key_decimal:
        seed = seed.bitwise_xor(i)
        
    torch.manual_seed(seed)
    _, perms = torch.rand((1,n)).to(device=device).squeeze().sort()
    
    if type == 'encryption':
         perm_img = img[:,perms]
    elif type == 'decryption' :
         perm_img = torch.zeros_like(img, device=device)
         perm_img[:,perms] = img
    
    return perm_img.to(torch.uint8)

def permutation_columns_rows(img : torch.Tensor, key_decimal : torch.Tensor, type :str,  device = 'cuda:0'):
    m,n = img.shape
    
    seedc = key_decimal[:16].sum()
    seedr = key_decimal[16:].sum() 
        
    torch.manual_seed(seedc)
    _, perms_c = torch.rand((1,n)).to(device=device).squeeze().sort()
    
    torch.manual_seed(seedr)
    _, perms_r = torch.rand((1,m)).to(device=device).squeeze().sort()
    
    if type == 'encryption':
         perm_img = img[perms_r,:]#[:,perms_c]
         perm_img = perm_img.clone()[:,perms_c]
    elif type == 'decryption' :
         perm_img = torch.zeros_like(img, device=device)
         perm_img[perms_r,:] = img
         perm_img[:,perms_c] = perm_img.clone()
    
    return perm_img.to(torch.uint8)  