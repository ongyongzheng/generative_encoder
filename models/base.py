import torch.nn as nn
import torch.nn.functional as F
import numpy as np

### BeGAN implementation
"""
Source: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/began/began.py
"""
class Generator(nn.Module):
    def __init__(self, img_size, latent_size, channels):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_size, 128 * self.init_size ** 2)
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Encoder(nn.Module):
    def __init__(self, img_size, latent_size, channels):
        super(Encoder, self).__init__()

        # down sampling
        self.down = nn.Sequential(
            nn.Conv2d(channels, 64, 3, 2, 1),
            nn.ReLU(),
        )
        self.down_size = img_size // 2
        down_dim = 64 * (img_size // 2) ** 2
        self.fc = nn.Sequential(
            nn.Linear(down_dim, latent_size),
            nn.BatchNorm1d(latent_size, 0.8),
            nn.Tanh(),
        )

    def forward(self, img):
        z = self.down(img)
        z = self.fc(z.view(z.size(0), -1))
        return z

class Decoder(nn.Module):
    def __init__(self, img_size, latent_size, channels):
        super(Decoder, self).__init__()

        self.down_size = img_size // 2
        down_dim = 64 * (img_size // 2) ** 2
        self.fc = nn.Sequential(
            nn.Linear(latent_size, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, channels, 3, 1, 1),
        )

    def forward(self, z):
        out = self.fc(z)
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
        return out

class Discriminator(nn.Module):
    def __init__(self, img_size, latent_size, channels):
        super(Discriminator, self).__init__()

        self.encoder = Encoder(img_size, latent_size, channels)
        self.decoder = Decoder(img_size, latent_size, channels)

    def forward(self, img):
        z = self.encoder(img)
        out = self.decoder(z)
        return out