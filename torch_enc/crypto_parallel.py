from time import time
import torch

import hashing.hashing as hashing
import torch_enc.crypto as crypto
from torch_enc.encoding import permutation_by_gcd, permutation_columns, permutation_columns_rows, permutation_rows

DECIMAL_LEN  = 128
HASHING_METHOD = hashing.SHA512

# def decrypt_parallel(enc_img : torch.Tensor, key_hex : str, hash_img, device = 'cuda:0',  chunk_m_step = 512, chunk_n_step = 512):
def decrypt_parallel(enc_img : torch.Tensor, key_hex : str,image_hash : str, device = 'cuda:0',  chunk_m_step = 512, chunk_n_step = 512):
    m,n = enc_img.shape
    t0 = time()
    key_hash = hashing.create_hash_key(image_hash, key_hex, meth=HASHING_METHOD)
    print(f'dec : create_hash_key : {time() - t0:.4f}')
    t0 = time()
    key_decimal = hashing.hex_str_to_decimal_tensor(hex_str=key_hash, decimal_len=DECIMAL_LEN, device=device)
    print(f'dec : hash_to_decimal :{time() - t0:.4f}')
    
    # dec_img = torch.zeros_like(enc_img)
    
    
    dec_img = permutation_columns_rows(enc_img, key_decimal, 'decryption', device)
    # dec_img = permutation_columns(enc_img, key_decimal, 'decryption', device)
    # dec_img = permutation_rows(dec_img, key_decimal, 'decryption', device)
    dec_img = permutation_by_gcd(dec_img, key_decimal, 'decryption',  device)

    
    for chunk_m in range(0, m, chunk_m_step): 
        for chunk_n in range(0, n, chunk_n_step):
            end_m_chunk = min(m,chunk_m+chunk_m_step)      
            end_n_chunk = min(n,chunk_n+chunk_n_step)  
            
            chunk_m_len = end_m_chunk - chunk_m    
            chunk_n_len = end_n_chunk - chunk_n   
            
            step_m = chunk_m // chunk_m_step
            step_n = chunk_n // chunk_n_step
                
            t0 = time()
            key_image = hashing.create_key_image(chunk_m_len, chunk_n_len,
                                                             key_decimal,
                                                             device)
            print(f'dec : create_key_image :{time() - t0:.4f}')
            t0 = time()
            dec_img[chunk_m:end_m_chunk, chunk_n:end_n_chunk] = crypto.decryption(dec_img[chunk_m:end_m_chunk, chunk_n:end_n_chunk],
                                                                                  key_image,
                                                                                  key_decimal,
                                                                                  100,
                                                                                  chunk_m_len, chunk_n_len,
                                                                                  step_m,
                                                                                  step_n,
                                                                                  device=device)
            print(f'dec : decryption :{time() - t0:.4f}')

    # dec_img = permutation_by_gcd(dec_img, key_decimal, 100, 'decryption',  device)

    return dec_img

def encrypt_parallel(img : torch.Tensor, key_hex : str, device = 'cuda:0', chunk_m_step = 512, chunk_n_step = 512):
    m,n = img.shape
    t0 = time()
    image_hash = hashing.hash_image(img, key_hex)
    print(f'enc : hash_image : {time() - t0:.4f}')
    t0 = time()
    key_hash = hashing.create_hash_key(image_hash, key_hex, meth=HASHING_METHOD)
    print(f'enc : create_hash_key : {time() - t0:.4f}')
    t0 = time()
    key_decimal = hashing.hex_str_to_decimal_tensor(hex_str=key_hash, decimal_len=DECIMAL_LEN, device=device)
    print(f'enc : hash_to_decimal :{time() - t0:.4f}')
    

    enc_img = torch.zeros_like(img)
    
    # enc_img = permutation_by_gcd(img, key_decimal, 100, 'encryption',  device)
    
    for chunk_m in range(0, m, chunk_m_step): 
        for chunk_n in range(0, n, chunk_n_step):
            end_m_chunk = min(m,chunk_m+chunk_m_step)      
            end_n_chunk = min(n,chunk_n+chunk_n_step)  
            
            chunk_m_len = end_m_chunk - chunk_m    
            chunk_n_len = end_n_chunk - chunk_n    
            
            step_m = chunk_m // chunk_m_step
            step_n = chunk_n // chunk_n_step
                
            t0 = time()
            key_image = hashing.create_key_image(chunk_m_len, chunk_n_len,
                                                             key_decimal,
                                                             device)
            print(f'enc : create_key_image :{time() - t0:.4f}')
            t0 = time()
            enc_img[chunk_m:end_m_chunk, chunk_n:end_n_chunk] = crypto.encryption(img[chunk_m:end_m_chunk, chunk_n:end_n_chunk],
                                                                                  key_image,
                                                                                  key_decimal,
                                                                                  100,
                                                                                  chunk_m_len, chunk_n_len,
                                                                                  step_m,
                                                                                  step_n,
                                                                                  device=device)
            print(f'enc : encryption :{time() - t0:.4f}')
    
    #perform big permutation
    enc_img = permutation_by_gcd(enc_img, key_decimal, 'encryption',  device)
    # enc_img = permutation_rows(enc_img, key_decimal, 'encryption', device)
    # enc_img = permutation_columns(enc_img, key_decimal, 'encryption', device)
    enc_img = permutation_columns_rows(enc_img, key_decimal, 'encryption', device)
    # enc_img = ((enc_img.int() + 10) % 256).to(torch.uint8)
    # return hash_img,enc_img
    return enc_img,image_hash