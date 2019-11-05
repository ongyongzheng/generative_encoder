import numpy as np
import torch

class Config(object):
    def __init__(self):
        # data configs
        self.dataset_path = './data/'
        self.save_path = './results/'
        self.mode = 'celeba'
        self.test_size = 0.1
        # nn configs
        self.img_size = 64
        self.latent_size = 62
        self.channels = 3 # num of image channels
        self.img_shape = (self.channels, self.img_size, self.img_size)
        self.num_workers = 2
        self.gamma = 0.5
        self.lambda_k = 0.001
        # training configs
        self.batch_size = 64
        self.random_size = 2*8
        self.n_epochs = 200
        self.lr = 0.0002
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")