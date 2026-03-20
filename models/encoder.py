import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),   # 128 -> 63
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),  # 63 -> 30
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), # 30 -> 14
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, latent_dim),
        )

    def forward(self, x):
        return self.net(x)
