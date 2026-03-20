import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64*8*8)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, 8, 8)
        return self.deconv(x)