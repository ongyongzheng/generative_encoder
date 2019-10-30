import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from progress.bar import Bar

from models.base import Generator, Discriminator

class BEGAN(object):
    def __init__(self, config):
        # save some variables
        self.img_size = config.img_size
        self.img_shape = config.img_shape
        self.lr = config.lr
        self.device = config.device
        self.batch_size = config.batch_size
        self.n_epochs = config.n_epochs
        self.latent_size = config.latent_size
        self.channels = config.channels

    def build_model(self):
        self.generator = Generator(self.img_size, self.latent_size, self.channels).to(self.device)
        self.discriminator = Discriminator(self.img_size, self.latent_size, self.channels).to(self.device)

    def train(self, train_data):
        bar = Bar('Training', max=len(train_data))
        for step, data in enumerate(train_data):
            bar.next()
        bar.finish()


    def test(self):
        pass

    def save_model(self, path):
        pass
