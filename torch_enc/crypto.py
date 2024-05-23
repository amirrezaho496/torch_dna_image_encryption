from time import time
import torch
import torch_enc.encoding as encoding

def encryption(plain_img: torch.Tensor, key_image: torch.Tensor, key_decimal: float, key_feature: int, m: int, n: int, step_m = 0, step_n = 0,  device = 'cuda:0') -> torch.Tensor:
    """
    Encrypts an image using a key image and decimal value.

    Args:
        plain_img: The original image to be encrypted
        key_image: The key image used for encryption
        key_decimal: A decimal value used in the encryption process
        key_feature: A feature value used in the encryption process
        m: The height of the image
        n: The width of the image

    Returns:
        The encrypted image
    """
    encoded_dif_img = encoding.encode_image_into_4_subcells(m, n, plain_img, device=device)
    encoded_dna_dif_img = encoding.encoded_image_into_dna_sequence(m, n, encoded_dif_img, key_decimal, key_feature, device=device)
    encoded_dna_per_image = encoding.permutation_dna(encoded_dna_dif_img, key_decimal, key_feature, m, n, 'encryption', step_m, step_n, device=device)
    dif_img_dna = encoding.diffusion_dna(encoded_dna_per_image, key_image, key_decimal, key_feature, m, n, 'encryption', step_m, step_n, device=device)
    encrypted_img = encoding.decoding_dna_image(m, n, dif_img_dna, key_decimal, key_feature, device=device)
    
    print(f'  encryption : ({m},{n})')
    return encrypted_img



def decryption(en_img, key_img, key_decimal, key_feature, m, n, step_m = 0, step_n = 0, device = 'cuda:0'):
    t0 = time()
    
    encoded_en_img = encoding.encode_image_into_4_subcells(m, n, en_img, device= device)
    encoded_dna_en_img = encoding.encoded_image_into_dna_sequence(m, n, encoded_en_img, key_decimal, key_feature, device= device)
    dif_img_dna = encoding.diffusion_dna(encoded_dna_en_img, key_img, key_decimal, key_feature, m, n, 'decryption', step_m, step_n, device=device)
    per_image_dna = encoding.permutation_dna(dif_img_dna, key_decimal, key_feature, m, n, 'decryption', step_m, step_n, device=device)    
    dec_image = encoding.decoding_dna_image(m, n, per_image_dna, key_decimal, key_feature, device=device)
    
    print(f'  decryption : ({m},{n})')
    
    return dec_image


