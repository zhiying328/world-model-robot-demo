import torch
import torch.nn as nn

class RSSM(nn.Module):
    def __init__(self, latent_dim=128, action_dim=3):
        super().__init__()

        self.rnn = nn.GRU(latent_dim + action_dim, latent_dim, batch_first=True)

        self.fc = nn.Linear(latent_dim, latent_dim)

    def forward(self, z, action, h):
        x = torch.cat([z, action], dim=-1).unsqueeze(1)
        out, h = self.rnn(x, h)
        z_next = self.fc(out.squeeze(1))
        return z_next, h