import torch
import torch.nn as nn
from torch import optim

from tqdm import tqdm
from diffusion.utils import expand_axis_like

class Diffusion:
    def __init__(self, noise_steps=1000, beta_0=1e-4, beta_T=1e-2, img_size=64, device="cuda") -> None:

        self.device = device

        self.noise_steps = 1000
        self.noise_steps = noise_steps

        self.beta = torch.linspace(beta_0, beta_T, noise_steps).to(self.device)
        self.alpha = 1 - self.beta  # formula (1)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)  # formula (2)

        self.img_size = img_size

    def noise_images(self, x, t):
        """
        x.dim = 4 (image index, rgb=3, img_H, img_W)
        """
        assert t < self.noise_steps, f"t must less than {self.noise_steps}"

        sqrt_alpha_hat = expand_axis_like(torch.sqrt(self.alpha_hat[t]), x)  # sqrt{\bar{\alpha}}
        sqrt_alpha_hat_complement = expand_axis_like(torch.sqrt(1.0 - self.alpha_hat[t]), x)  # sqrt{1 - \bar{\alpha}}

        epsilon = torch.randn_like(x)  # formula (3) condition; epsilon ~ N(0, I)

        # formula (3)
        return sqrt_alpha_hat * x + sqrt_alpha_hat_complement * epsilon, epsilon

    def sample_timesteps(self, t):
        return torch.randint(low = 1, high = self.noise_steps, size = (n,))
    

    # Algorithm (2)
    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(range(self.noise_steps - 1, 0, -1), position=0):
                t = torch.full(n, i).long().to(self.device)
                parameterized_epsilon = model(x, t) # predicted_noise

                alpha = expand_axis_like(self.alpha[t], x)
                alpha_hat = expand_axis_like(self.alpha_hat[t], x)
                beta = expand_axis_like(self.beta[t], x)

                z = torch.rand_like(x) if i > 1 else torch.zeros_like(x)

                # Algorithm (2) line 4:
                # We don't predict variance. We assume that large noise_time_step the variance(\beta) will converge to its upper bound
                x = 1 / torch.sqrt(alpha) * (x - (1-alpha) / torch.sqrt(1-alpha_hat) * parameterized_epsilon) + torch.sqrt(beta) * z
        model.train()
        # x = x_0
        x = (x.clamp(-1, 1) + 1) / 2 # shift x from [-1, 1] to [0, 1]
        x = (x * 255).type(torch.uint8)
        return x