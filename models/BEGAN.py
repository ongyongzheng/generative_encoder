import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from models.base import Generator, Discriminator

class BEGAN(object):
    def __init__(self, config):
        self.generator = Generator(config.img_size, config.latent_size, config.channels)
        self.discriminator = Discriminator(config.img_size, config.latent_size, config.channels)
        # save some variables
        self.img_size = config.img_size
        self.img_shape = config.img_shape
        self.lr = config.lr
        self.device = config.device
        self.batch_size = config.batch_size
        self.n_epochs = config.n_epochs

    def train(self):
        pass

    def test(self):
        pass

    def save_model(self, path):
        pass
