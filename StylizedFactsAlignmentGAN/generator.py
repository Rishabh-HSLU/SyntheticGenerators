import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim: int, T: int, n_assets: int, hidden_dim: int = 256):
        super().__init__()
        self.T = T
        self.n_assets = n_assets


        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim*2, T*n_assets),
        )

    def forward(self, z:torch.Tensor):
        out = self.net(z)
        return out.view(-1, self.T, self.n_assets)
    

