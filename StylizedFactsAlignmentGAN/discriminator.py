import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, T: int, n_assets: int, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(T * n_assets, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, 1)
        )

    def forward(self, x: torch.Tensor):
            x = x.view(x.size(0), -1)
            return self.net(x)
