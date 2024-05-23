import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import tools

from time import time
from PIL import Image

from torch_enc.crypto_parallel import decrypt_parallel, encrypt_parallel

def display_images(original_img : torch.Tensor, encrypted_img : torch.Tensor, decrypted_img : torch.Tensor):    
    
    # fig, axs = plt.subplots(1, 3, figsize=(15, 10))
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # Display images
    imgs = [original_img, encrypted_img, decrypted_img]
    titles = ['Original Image', 'Encrypted Image', 'Decrypted Image']
    for i, img in enumerate(imgs):
        axs[0, i].imshow(img.cpu(), cmap='gray')
        axs[0, i].set_title(titles[i])

    # Display histograms
    for i, img in enumerate(imgs):
        histogram = torch.histogram(img.reshape(-1).float().cpu(), bins=256)
        hist = histogram.hist
        bins = histogram.bin_edges
        axs[1, i].bar(range(0,256), hist.flatten().cpu(), color='gray', alpha=0.7)
        axs[1, i].set_title('Histogram of ' + titles[i])

    plt.tight_layout()
    plt.show()
# Main execution

def main():
    torch.rand(1000)
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cuda:0'
    # device = 'cpu'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Read the image and convert it to grayscale
    
    ## Open the image file
    img = Image.open('imgs/12k.jpg').convert('L')
    # img = Image.open('imgs/8k.jpg').convert('L')
    # img = Image.open('imgs/cat-4k.jpg').convert('L')
    # img = Image.open('imgs/pixabay-FHD.jpg').convert('L')
    # img = Image.open('imgs/Lena512.bmp').convert('L')
    
    ## Convert the Image object to a numpy array
    img = np.array(img)
    img = torch.from_numpy(img).to(device)

    m, n = img.shape


    # Set the key and crop parameters
    key_hex = 'amirreza_khc_ui_aashbduygskdaskhd'

    crop = 4
    noise_level = 0.00


    # Add noise to the image

    noise = torch.rand(m, n, device=device) * noise_level * 255

    # img = img + noise

    chunk_m_step = 2048
    chunk_n_step = 2048
    
    itr_num = 1
    enc_times = []
    for itr in range(itr_num):
        # Encrypt the image
        enc_time0 = time()

        
        # hash_img,enc_img = encrypt_parallel(img, key_hex, device, chunk_m_step, chunk_n_step)
        enc_img, image_hash = encrypt_parallel(img, key_hex, device, chunk_m_step, chunk_n_step)
                
        t1 = time()
        # Print the encryption time, entropy, and correlation coefficient
        enc_times.append(t1 - enc_time0)
        print(f'Encryption Time = {t1 - enc_time0:.4f}\n')

    print("--------------------------------------------------------------")
    
    
    # Adding noise to enc image :
    # enc_img = enc_img + noise
    
    dec_times = []
    for itr in range(itr_num):    
        dec_time0 = time()
        # dec_img = decrypt_parallel(enc_img, key_hex, hash_img, device, chunk_m_step, chunk_n_step)
        dec_img = decrypt_parallel(enc_img, key_hex, image_hash, device, chunk_m_step, chunk_n_step)
        
        t1 = time()
        dec_times.append(t1 - dec_time0)
        print(f'Decryption Time = {t1- dec_time0:.4f}\n') 

    print("--------------------------------------------------------------")
    avg_enc_times = np.array(enc_times).mean()
    print(f"last enc times : {enc_times[-1]:.4f}")      
    print(f"avrage enc times : {avg_enc_times:.4f}\n")      
    avg_dec_times = np.array(dec_times).mean()
    print(f"last dec times : {dec_times[-1]:.4f}")      
    print(f"avrage dec times : {avg_dec_times:.4f}")    
    print("--------------------------------------------------------------\n")

    
    defrent = torch.sum(img - dec_img)    
    print (f'Img - dec_img : {defrent}')
    
    uaci , npcr = tools.uaci_npcr(enc_img, img)
    print (f'UACI : {uaci:.4f}, NPCR : {npcr:.4f}')
    
    pimg_entropy = tools.image_entropy(img, n)
    encimg_entropy = tools.image_entropy(enc_img, n)
    print (f'Plain image Entropy     : {pimg_entropy} \nEncrypted image Entropy : {encimg_entropy}')
    
    cc, x,y = tools.adjancy_corr_pixel_rand(img, enc_img)
    print("corrcoef : ")
    print(f"Vertical   : plain inage :{cc[0][1]:.4f}, encrypted image : {cc[0][2]:.4f}")
    print(f"Horizontal : plain inage :{cc[1][1]:.4f}, encrypted image : {cc[1][2]:.4f}")
    print(f"Diagonal   : plain inage :{cc[2][1]:.4f}, encrypted image : {cc[2][2]:.4f}")
    print("--------------------------------------------------------------\n")
    display_images(img, enc_img, dec_img)




if __name__ == '__main__':

    main()
    pass

