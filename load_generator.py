import torch
from trainer.generator.generator import Generator_CIFAR
import numpy as np
from PIL import Image


torch.load('./_3.pth')
g = Generator_CIFAR()
g.to(0)
y_input = torch.tensor([52,52,52,52,52], dtype=torch.long).to(0)
diffs = torch.tensor([1,3,5,7,9], dtype=torch.long).to(0)
eps = torch.rand((y_input.shape[0], g.noise_dim)).to(0)

with torch.no_grad():
    imgs = g(diffs, y_input, eps, raw=True)
    for i, img in enumerate(imgs):
        array = np.transpose(img.cpu().numpy(), (1, 2, 0))
        img = Image.fromarray(array.astype(np.uint8))
        img.save(f'generators/dlevel_{diffs[i].cpu().item()}_l_{y_input[i].cpu().item()}.png')