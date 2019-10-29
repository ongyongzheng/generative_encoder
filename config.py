import numpy as np
import torch

class Config(object):
    def __init__(self):
        self.dataset_path = './data/'
        self.save_path = './results/'
        self.mode = 'celeba'
        # nn configs
        self.img_size = 64
        self.latent_size = 62
        self.channels = 1 # num of image channels
        self.img_shape = (self.channels, self.img_size, self.img_size)
        # training configs
        self.batch_size = 64
        self.n_epochs = 200
        self.lr = 0.0002
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")