import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)