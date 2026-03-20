import torch
import numpy as np
from PIL import Image

from models.encoder import Encoder
from models.rssm import RSSM
from models.decoder import Decoder


class WorldModel:
    def __init__(self, device="cpu"):
        self.device = device
        self.encoder = Encoder().to(device)
        self.rssm = RSSM().to(device)
        self.decoder = Decoder().to(device)

        self.encoder.eval()
        self.rssm.eval()
        self.decoder.eval()

        self._h = torch.zeros(1, 1, 128).to(device)
        self._z = None

    def reset(self):
        self._h = torch.zeros(1, 1, 128).to(self.device)
        self._z = None

    def predict(self, obs, action):
        """
        obs:    np.array [H, W, 3] uint8
        action: np.array [3]
        Returns: (z_next tensor [1, 128], imagined_obs np.array [128, 128, 3] uint8)
        """
        with torch.no_grad():
            obs_t = torch.tensor(obs / 255.0, dtype=torch.float32)
            obs_t = obs_t.permute(2, 0, 1).unsqueeze(0).to(self.device)  # [1, 3, H, W]
            z = self.encoder(obs_t)  # [1, 128]

            action_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, 3]
            z_next, self._h = self.rssm(z, action_t, self._h)  # [1, 128]

            recon = self.decoder(z_next)  # [1, 3, H', W']
            recon_np = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
            recon_np = (recon_np * 255).clip(0, 255).astype(np.uint8)

            # Resize to 128x128 to match real obs
            img = Image.fromarray(recon_np)
            img = img.resize((128, 128), Image.BILINEAR)
            imagined_obs = np.array(img)

            self._z = z_next

        return z_next, imagined_obs
