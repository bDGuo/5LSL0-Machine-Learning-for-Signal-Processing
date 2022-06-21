import torch

def re_kspace(p_kspace):
    out = torch.zeros_like(p_kspace)
    for i in range(len(p_kspace)):
        out[i] = torch.fft.ifftshift(p_kspace[i])
    return torch.fft.ifft2(out)