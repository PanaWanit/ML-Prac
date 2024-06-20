import torch
from Diffusion.diffusion.ddpm import Diffusion

model = Diffusion(device="cpu")

x = torch.randn(3, 100, 100, 100)

res = model.noise_images(x, 10)
print('The result of noise image is')
print(res)