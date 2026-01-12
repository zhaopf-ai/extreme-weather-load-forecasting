import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gaussian_kernel(y, sigma=0.5):
    diff = y.unsqueeze(0) - y.unsqueeze(1)
    dist_squared = torch.sum(diff ** 2, dim=-1)
    kernel = torch.exp(-dist_squared / (2 * sigma ** 2))
    return kernel / kernel.sum(dim=1, keepdim=True)


def c_mixup(x1, x2, x3, y, alpha=0.4, sigma=0.5):
    batch_size = x1.size(0)
    kernel = gaussian_kernel(y, sigma=sigma)
    indices = torch.multinomial(kernel, num_samples=1).squeeze(1)
    x1_j = x1[indices]
    x2_j = x2[indices]
    x3_j = x3[indices]
    y_j = y[indices]
    lam = torch.distributions.Beta(alpha, alpha).sample([batch_size, 1])
    lam = lam.to(device)

    x1_mix = lam * x1 + (1 - lam) * x1_j
    x2_mix = lam * x2 + (1 - lam) * x2_j
    x3_mix = lam * x3 + (1 - lam) * x3_j
    y_mix = lam * y + (1 - lam) * y_j

    return x1_mix, x2_mix, x3_mix, y_mix, indices, lam
