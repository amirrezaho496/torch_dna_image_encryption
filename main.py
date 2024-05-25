import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import tools

from time import time
from PIL import Image

from torch_enc.crypto_parallel import decrypt_parallel, encrypt_parallel

def display_images(original_img : torch.Tensor, encrypted_img : torch.Tensor, decrypted_img : torch.Tensor, subtitle = "Image Encryption"):    
    
    # fig, axs = plt.subplots(1, 3, figsize=(15, 10))
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # Display images
    imgs = [original_img, encrypted_img, decrypted_img]
    titles = ['Original Image', 'Encrypted Image', 'Decrypted Image']
    for i, img in enumerate(imgs):
        axs[0, i].imshow(img.cpu(), cmap='gray')
        axs[0, i].set_title(titles[i])

    # Display histograms
    max_y_hist = 0
    for i, img in enumerate(imgs):
        histogram = torch.histogram(img.reshape(-1).float().cpu(), bins=256)
        hist = histogram.hist
        max_y_hist = max(hist.max().item(), max_y_hist)
        # bins = histogram.bin_edges
        axs[1, i].plot(range(0,256), hist.flatten().cpu())
        axs[1, i].set_title(f'Histogram of {titles[i]}')
        axs[1, i].set_ylim(bottom=0, top = max_y_hist)
        
    fig.suptitle(subtitle, fontsize=16)
    plt.tight_layout()
    plt.show()
# Main execution

def cropping_attack(img : torch.Tensor, enc_img : torch.Tensor, key, image_hash, device, chunk_m_step, chunk_n_step):
    m,n = img.shape
    ranges = [1.5,2,4,8,16]
    enc_croped_imgs = []
    dec_croped_imgs = []
    titles = []
    for i in ranges:
        crop_div = np.sqrt(i)
        cr_m = int(m/crop_div)
        cr_n = int(n/crop_div)

        enc_croped_img = enc_img.clone()
        enc_croped_img[:cr_m,:cr_n] = 0
        enc_croped_imgs.append(enc_croped_img)
        
        dec_croped_img = decrypt_parallel(enc_croped_img, key, image_hash, device, chunk_m_step, chunk_n_step)
        psnr = tools.psnr(img,dec_croped_img)
        print(f'PSNR decrypt cropped image 1/{crop_div*crop_div :.2f} : {psnr:.4f} ')
        dec_croped_imgs.append(dec_croped_img)
        titles.append(f'cropped 1/{crop_div*crop_div :.1f}, PNSR : {psnr:.4f} ')
        pass
    
    fig, axs = plt.subplots(2, len(ranges), figsize=(15, 10))

    
    for i, cr_img in enumerate(enc_croped_imgs):
        axs[0, i].imshow(cr_img.cpu(), cmap='gray')
        axs[0, i].set_title(f'cropped 1/{ranges[i]}')
    for i, dec_cr_img in enumerate(dec_croped_imgs):
        axs[1, i].imshow(dec_cr_img.cpu(), cmap='gray')
        axs[1, i].set_title(titles[i])
       
    fig.suptitle("Cropping attack", fontsize=16) 
    
    plt.tight_layout()
    plt.show()
    pass

def salt_pepper_noise_attack(img : torch.Tensor, enc_img : torch.Tensor, key, image_hash, device, chunk_m_step, chunk_n_step):
    m,n = img.shape
    ranges = [660,500,250,125,62]
    enc_nsy_imgs = []
    dec_nsy_imgs = []
    titles = []
    for i in ranges:
        noise_level = 0.001 * i 
        
        enc_nsy_img = tools.salt_pepper_noise(enc_img, noise_level)
        enc_nsy_imgs.append(enc_nsy_img)
        
        dec_nsy_img = decrypt_parallel(enc_nsy_img, key, image_hash, device, chunk_m_step, chunk_n_step)
        psnr = tools.psnr(img,dec_nsy_img)
        print(f'PSNR decrypt noisy image {noise_level:.3f} : {psnr:.4f} ')
        dec_nsy_imgs.append(dec_nsy_img)
        titles.append(f'noise {noise_level:.3f}, PNSR : {psnr:.4f} ')
        pass
    
    fig, axs = plt.subplots(2, len(ranges), figsize=(15, 10))

    
    for i, nsy_img in enumerate(enc_nsy_imgs):
        axs[0, i].imshow(nsy_img.cpu(), cmap='gray')
        axs[0, i].set_title(f'noise {0.001 * ranges[i]:.3f}')
    for i, dec_nsy_img in enumerate(dec_nsy_imgs):
        axs[1, i].imshow(dec_nsy_img.cpu(), cmap='gray')
        axs[1, i].set_title(titles[i])
       
    fig.suptitle('Salt and pepper noise attack', fontsize=16) 
    
    plt.tight_layout()
    plt.show()
    pass

def some_bit_change_attack(img : torch.Tensor, enc_img : torch.Tensor, key, image_hash, device, chunk_m_step, chunk_n_step):
    m,n = img.shape
    ranges = [1,2,4,8,16]
    enc_nsy_imgs = []
    dec_nsy_imgs = []
    titles = []
    for i in ranges:
        rands = torch.rand(i, device=device)
        ms = torch.floor(rands* m).int()
        ns = torch.floor(rands* n).int()
        
        enc_nsy_img = enc_img
        enc_nsy_img[ms,ns] += 1
        enc_nsy_imgs.append(enc_nsy_img)
        
        dec_nsy_img = decrypt_parallel(enc_nsy_img, key, image_hash, device, chunk_m_step, chunk_n_step)
        psnr = tools.psnr(img,dec_nsy_img)
        print(f'PSNR decrypt {i} bit chnage image : {psnr:.4f} ')
        dec_nsy_imgs.append(dec_nsy_img)
        titles.append(f'{i} change image, PNSR : {psnr:.4f} ')
        pass
    
    fig, axs = plt.subplots(2, len(ranges), figsize=(15, 10))

    
    for i, nsy_img in enumerate(enc_nsy_imgs):
        axs[0, i].imshow(nsy_img.cpu(), cmap='gray')
        axs[0, i].set_title(f'{ranges[i]} bit change image')
    for i, dec_nsy_img in enumerate(dec_nsy_imgs):
        axs[1, i].imshow(dec_nsy_img.cpu(), cmap='gray')
        axs[1, i].set_title(titles[i])
       
    fig.suptitle('Change some bit in encrypted image', fontsize=16) 
    
    plt.tight_layout()
    plt.show()
    pass


    

def main():
    torch.rand(1000)
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cuda:0'
    # device = 'cpu'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Read the image and convert it to grayscale
    
    ## Open the image file
    # img = Image.open('imgs/12k.jpg').convert('L')
    # img = Image.open('imgs/8k.jpg').convert('L')
    # img = Image.open('imgs/cat-4k.jpg').convert('L')
    # img = Image.open('imgs/pixabay-FHD.jpg').convert('L')
    img = Image.open('imgs/Lena512.bmp').convert('L')
    
    ## Convert the Image object to a numpy array
    img = np.array(img)
    img = torch.from_numpy(img).to(device)
    
    # For making image of any sizes :
    # size = 11_000
    # img = torch.arange(end = size*size, device=device).reshape(size, size)
    # img = img / (size*size)
    # img *= 255
    # img = img.int()
    

    m, n = img.shape


    # Set the key and crop parameters
    key = '123456789'

    crop = 4
    noise_level = 0.00


    # Add noise to the image

    noise = torch.rand(m, n, device=device) * noise_level * 255

    # img = img + noise

    chunk_m_step = 4000
    chunk_n_step = 4000
    
    itr_num = 1
    enc_times = []
    for itr in range(itr_num):
        # Encrypt the image
        enc_time0 = time()

        
        # hash_img,enc_img = encrypt_parallel(img, key_hex, device, chunk_m_step, chunk_n_step)
        enc_img, image_hash = encrypt_parallel(img, key, device, chunk_m_step, chunk_n_step)
                
        t1 = time()
        # Print the encryption time, entropy, and correlation coefficient
        enc_times.append(t1 - enc_time0)
        print(f'Encryption Time = {t1 - enc_time0:.4f}\n')

    print("--------------------------------------------------------------")
    
    dec_times = []
    for itr in range(itr_num):    
        dec_time0 = time()
        # dec_img = decrypt_parallel(enc_img, key_hex, hash_img, device, chunk_m_step, chunk_n_step)
        dec_img = decrypt_parallel(enc_img, key, image_hash, device, chunk_m_step, chunk_n_step)
        
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
    
    

    
    defrent = torch.sum(torch.abs(img - dec_img))    
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
    
    print("--------------------------------------------------------------")
    print('Cropping attack : \n')
    
    cropping_attack(img,enc_img, key, image_hash, device, chunk_m_step, chunk_n_step)
    
    
    print("--------------------------------------------------------------")
    print('Salt and pepper noise attack : \n')
    salt_pepper_noise_attack(img,enc_img, key, image_hash, device, chunk_m_step, chunk_n_step)

    print("--------------------------------------------------------------")
    print('Some bit change attack : \n')
    some_bit_change_attack(img,enc_img, key, image_hash, device, chunk_m_step, chunk_n_step)

    

if __name__ == '__main__':

    main()
    pass

