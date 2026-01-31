import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            self._conv_block(1, 16),   
            self._conv_block(16, 32), 
            self._conv_block(32, 64),  
            self._conv_block(64, 128), 
            nn.Flatten(),
            nn.Linear(128 * 15 * 15, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 15 * 15),
            nn.Unflatten(1, (128, 15, 15)),
            self._deconv_block(128, 64),
            self._deconv_block(64, 32),
            self._deconv_block(32, 16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def _conv_block(self, in_f, out_f):
        return nn.Sequential(
            nn.Conv2d(in_f, out_f, 3, stride=2, padding=1),
            nn.ReLU()
        )

    def _deconv_block(self, in_f, out_f):
        return nn.Sequential(
            nn.ConvTranspose2d(in_f, out_f, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z