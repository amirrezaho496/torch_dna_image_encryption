import torch
import numpy as np 

def uaci_npcr(c1 : torch.Tensor, c2 : torch.Tensor):
    m, n = c1.shape
    c1 = c1.to(torch.float)
    c2 = c2.to(torch.float)
    
    # h = lambda i, j: 0 if c2[i, j] == c1[i, j] else 1
    # e = lambda i, j: abs(c2[i, j] - c1[i, j])
    # uaci = sum(e(i, j) for j in range(n) for i in range(m)) / (255 * m * n)
    # npcr = sum(h(i, j) for j in range(n) for i in range(m)) / (m * n)
    
    uaci = torch.abs(c1 - c2) / 255
    uaci = uaci.mean().item()
    
    npcr = (c1 != c2).float().mean().item()
    
    return 100 * uaci, 100 * npcr

def image_entropy(img : torch.Tensor, n):
    hist = torch.histogram(img.reshape(-1).float().cpu(), bins=n).hist
    pdf = hist / torch.sum(hist)
    nonzero_indices = torch.nonzero(pdf).squeeze()
    entropy_val = -torch.sum(pdf[nonzero_indices] * torch.log2(pdf[nonzero_indices]))
    return entropy_val.item()

def salt_pepper_noise(image : torch.Tensor, prob : torch.float):
    output = image.clone()
    rand = torch.rand_like(image.float())
    
    output[rand < (prob/2)] = 0
    output[rand > (1 - (prob/2))] = 255
    
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         rdn = np.random.rand()
    #         if rdn < prob:
    #             output[i][j] = 0
    #         elif rdn > (1 - prob):
    #             output[i][j] = 255
    #         else:
    #             output[i][j] = image[i][j]
    return output


def psnr(original:torch.Tensor, compressed : torch.Tensor):
    max_pixel = 255
    mse = torch.mean((original.float() - compressed.float()) ** 2)
    if mse == 0:
        return 100
    return 10 * torch.log10((max_pixel**2) / mse)


def adjancy_corr_pixel_rand(plain_img : torch.Tensor, enc_img : torch.Tensor):
    plain_img = plain_img.to(torch.float)
    enc_img = enc_img.to(torch.float)
    m, n = plain_img.shape
    m -= 1
    n -= 1
    k = 100
    s = torch.randperm(m*n)[:k]
    x, y = torch.unravel_index(s, (m, n))
    # x = torch.from_numpy(x)
    # y = torch.from_numpy(y)
    return (
        [
            [
                "V",
                np.corrcoef(plain_img[x, y].cpu(), plain_img[x, y + 1].cpu())[0, 1],
                np.corrcoef(enc_img[x, y].cpu(), enc_img[x, y + 1].cpu())[0, 1],
            ],
            [
                "H",
                np.corrcoef(plain_img[x, y].cpu(), plain_img[x + 1, y].cpu())[0, 1],
                np.corrcoef(enc_img[x, y].cpu(), enc_img[x + 1, y].cpu())[0, 1],
            ],
            [
                "D",
                np.corrcoef(plain_img[x, y].cpu(), plain_img[x + 1, y + 1].cpu())[0, 1],
                np.corrcoef(enc_img[x, y].cpu(), enc_img[x + 1, y + 1].cpu())[0, 1],
            ],
        ],
        x,
        y,
    )