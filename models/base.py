from math import log2

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

### BeGAN implementation
"""
Source:
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/began/began.py
https://github.com/1Konny/BEGAN-pytorch
"""
"""class Generator(nn.Module):
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

    def save_model(self, path):
        torch.save(self.state_dict(), path)"""

"""class Encoder(nn.Module):
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

    def save_model(self, path):
        torch.save(self.state_dict(), path)

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
            nn.Tanh()
        )

    def forward(self, z):
        out = self.fc(z)
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
        return out

    def save_model(self, path):
        torch.save(self.state_dict(), path)"""

def base_decoder_block(_type, n_filter=128, n_repeat=2):
    layers = []
    if _type == 'front':
        for i in range(n_repeat):
            layers.append(nn.Conv2d(n_filter, n_filter, 3, 1, 1))
            layers.append(nn.ELU(True))
            layers.append(nn.Conv2d(n_filter, n_filter, 3, 1, 1))
            layers.append(nn.ELU(True))

    elif _type == 'inter':
        for i in range(n_repeat):
            if i == 0:
                layers.append(nn.Conv2d(2*n_filter, n_filter, 3, 1, 1))
            else:
                layers.append(nn.Conv2d(n_filter, n_filter, 3, 1, 1))
            layers.append(nn.ELU(True))
            layers.append(nn.Conv2d(n_filter, n_filter, 3, 1, 1))
            layers.append(nn.ELU(True))

    elif _type == 'end':
        for i in range(n_repeat):
            if i != (n_repeat-1):
                layers.append(nn.Conv2d(n_filter, n_filter, 3, 1, 1))
                layers.append(nn.ELU(True))
            else:
                layers.append(nn.Conv2d(n_filter, 3, 3, 1, 1))
                #layers.append(nn.Tanh())
                #layers.append(nn.Sigmoid())

    else:
        raise

    return layers


def base_encoder_block(_type, n_filter=128, n_repeat=2, inter_scale=1):
    m = inter_scale

    layers = []
    if _type == 'front':
        for i in range(n_repeat):
            if i == 0:
                layers.append(nn.Conv2d(3, n_filter, 3, 1, 1))
                layers.append(nn.ELU(True))
            else:
                layers.append(nn.Conv2d(n_filter, n_filter, 3, 1, 1))
                layers.append(nn.ELU(True))

    elif _type == 'inter':
        for i in range(n_repeat):
            layers.append(nn.Conv2d(m*n_filter, m*n_filter, 3, 1, 1))
            layers.append(nn.ELU(True))
            if i != (n_repeat-1):
                layers.append(nn.Conv2d(m*n_filter, m*n_filter, 3, 1, 1))
                layers.append(nn.ELU(True))
            else:
                layers.append(nn.Conv2d(m*n_filter, (m+1)*n_filter, 3, 2, 1))
                layers.append(nn.ELU(True))

    elif _type == 'end':
        for i in range(n_repeat):
            layers.append(nn.Conv2d(m*n_filter, m*n_filter, 3, 1, 1))
            layers.append(nn.ELU(True))
            layers.append(nn.Conv2d(m*n_filter, m*n_filter, 3, 1, 1))
            layers.append(nn.ELU(True))

    else:
        raise

    return layers


class Decoder(nn.Module):
    def __init__(self, image_size, hidden_dim=64, n_filter=128, n_repeat=4):
        super(Decoder, self).__init__()
        self.image_size = image_size
        self.n_upsample = int(log2(image_size//8))
        self.hidden_dim = hidden_dim
        self.n_filter = n_filter
        self.n_repeat = n_repeat

        self.fc = nn.Linear(self.hidden_dim, 8*8*self.n_filter)
        self.convs = dict()
        for i in range(self.n_upsample+2):
            if i == 0:
                self.convs[i] = nn.Sequential(*base_decoder_block('front', n_filter, n_repeat))
                self.add_module(name='front', module=self.convs[i])
            elif i <= self.n_upsample:
                self.convs[i] = nn.Sequential(*base_decoder_block('inter', n_filter, n_repeat))
                self.add_module(name='inter'+str(i), module=self.convs[i])
            else:
                self.convs[i] = nn.Sequential(*base_decoder_block('end', n_filter, n_repeat))
                self.add_module(name='end', module=self.convs[i])


    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, h):
        h0 = self.fc(h)
        h0 = h0.view(h0.size(0), self.n_filter, 8, 8)

        out = self.convs[0](h0)
        out = torch.cat([out, h0], dim=1)
        out = F.upsample(out, scale_factor=2, mode='nearest')

        for i in range(1, self.n_upsample):
            out = self.convs[i](out)
            h0 = F.upsample(h0, scale_factor=2, mode='nearest')
            out = torch.cat([out, h0], dim=1)
            out = F.upsample(out, scale_factor=2, mode='nearest')

        out = self.convs[i+1](out)
        out = self.convs[i+2](out)

        return out

    def save_model(self, path):
        torch.save(self.state_dict(), path)

class Encoder(nn.Module):
    def __init__(self, image_size, hidden_dim=64, n_filter=128, n_repeat=2):
        super(Encoder, self).__init__()
        self.image_size = image_size
        self.n_upsample = int(log2(self.image_size//8))
        self.hidden_dim = hidden_dim
        self.n_filter = n_filter
        self.n_repeat = n_repeat

        self.convs = dict()
        for i in range(self.n_upsample+2):
            if i == 0:
                self.convs[i] = nn.Sequential(*base_encoder_block('front', self.n_filter, self.n_repeat))
                self.add_module('front', self.convs[i])
            elif i <= self.n_upsample:
                self.convs[i] = nn.Sequential(*base_encoder_block('inter', self.n_filter, self.n_repeat, i))
                self.add_module('inter'+str(i), self.convs[i])
            else:
                self.convs[i] = nn.Sequential(*base_encoder_block('end', self.n_filter, self.n_repeat, i))
                self.add_module('end', self.convs[i])

        self.fc = nn.Linear(8*8*i*self.n_filter, self.hidden_dim)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, image):
        out = self.convs[0](image)
        for i in range(1, len(self.convs.keys())):
            out = self.convs[i](out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        #out = nn.Tanh()(out)

        return out

    def save_model(self, path):
        torch.save(self.state_dict(), path)

def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()

class Discriminator(nn.Module):
    def __init__(self, img_size):
        super(Discriminator, self).__init__()

        self.encoder = Encoder(img_size, n_repeat=2)
        self.decoder = Decoder(img_size, n_repeat=2)

    def forward(self, img):
        z = self.encoder(img)
        out = self.decoder(z)
        return out

    def weight_init(self, mean, std):
        self.encoder.weight_init(mean, std)
        self.decoder.weight_init(mean, std)

class Generator(nn.Module):
    def __init__(self, img_size):
        super(Generator, self).__init__()
        self.decoder = Decoder(img_size, n_repeat=2)

    def forward(self, h):
        out = self.decoder(h)

        return out

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def weight_init(self, mean, std):
        self.decoder.weight_init(mean, std)

class ConvAE(nn.Module):
    def __init__(self, img_size=64, hidden_dim=128):
        super(ConvAE, self).__init__()
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        f = 64
        self.f = f # based on Chen Lin's code
        k = 3
        p = 0
        s = 2
        c_p = 1

        # prepare ConvAE
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, f, k, padding=c_p),
            nn.ReLU(),
            nn.Conv2d(f, f, k, padding=c_p),
            nn.ReLU(),
            nn.Conv2d(f, f, k, padding=c_p),
            nn.ReLU(),
            nn.Conv2d(f, 2*f, k, padding=c_p),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=s, padding=p)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2*f, 2*f, k, padding=c_p),
            nn.ReLU(),
            nn.Conv2d(2*f, 2*f, k, padding=c_p),
            nn.ReLU(),
            nn.Conv2d(2*f, 3*f, k, padding=c_p),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=s, padding=p)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(3*f, 3*f, k, padding=c_p),
            nn.ReLU(),
            nn.Conv2d(3*f, 3*f, k, padding=c_p),
            nn.ReLU(),
            nn.Conv2d(3*f, 4*f, k, padding=c_p),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=s, padding=p)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(4*f, 4*f, k, padding=c_p),
            nn.ReLU(),
            nn.Conv2d(4*f, 4*f, k, padding=c_p),
            nn.ReLU()
        )
        self.conv4_128 = nn.Sequential(
            nn.Conv2d(4*f, 5*f, k, padding=c_p),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=s, padding=p),
            nn.Conv2d(5*f, 5*f, k, padding=c_p),
            nn.ReLU(),
            nn.Conv2d(5*f, 5*f, k, padding=c_p),
            nn.ReLU()
        )
        self.num_pool = 3
        self.num_fil = 4
        if img_size == 128:
            self.num_pool += 1
            self.num_fil += 1
        out_size = int(img_size / (2 ** self.num_pool))
        print("Out Size: {}".format(out_size))
        self.lin_size = out_size * out_size * self.num_fil * f
        self.fc1 = nn.Linear(self.lin_size, 8*4*self.hidden_dim)
        self.feature = nn.Linear(8*4*self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, 8*4*self.hidden_dim)
        self.fc4 = nn.Linear(8*4*self.hidden_dim, self.lin_size)
        self.deconv4_128 = nn.Sequential(
            nn.ConvTranspose2d(5*f, 5*f, k, padding=c_p),
            nn.ReLU(),
            nn.ConvTranspose2d(5*f, 5*f, k, padding=c_p),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ConvTranspose2d(5*f, 4*f, k, padding=c_p),
            nn.ReLU()
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(4*f, 4*f, k, padding=c_p),
            nn.ReLU(),
            nn.ConvTranspose2d(4*f, 4*f, k, padding=c_p),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ConvTranspose2d(4*f, 3*f, k, padding=c_p),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(3*f, 3*f, k, padding=c_p),
            nn.ReLU(),
            nn.ConvTranspose2d(3*f, 3*f, k, padding=c_p),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ConvTranspose2d(3*f, 2*f, k, padding=c_p),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(2*f, 2*f, k, padding=c_p),
            nn.ReLU(),
            nn.ConvTranspose2d(2*f, 2*f, k, padding=c_p),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ConvTranspose2d(2*f, f, k, padding=c_p),
            nn.ReLU()
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(f, f, k, padding=c_p),
            nn.ReLU(),
            nn.ConvTranspose2d(f, 3, k, padding=c_p)
        )

    def encode(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if self.img_size == 128:
            x = self.conv4_128(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        out = self.feature(x)
        return out

    def decode(self, z):
        z = self.fc3(z)
        z = self.fc4(z)
        z = z.view(z.size(0),self.num_fil*self.f,8,8)
        if self.img_size == 128:
            z = self.deconv4_128(z)
        z = self.deconv4(z)
        z = self.deconv3(z)
        z = self.deconv2(z)
        out = self.deconv1(z)
        return out

    def forward(self, input):
        z = self.encode(input)
        out = self.decode(z)
        return out

    def save_model(self, path):
        torch.save(self.state_dict(), path)

