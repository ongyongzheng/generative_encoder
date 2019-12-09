import numpy as np
import torch

class Config(object):
    def __init__(self):
        # data configs
        self.dataset_path = './data/'
        self.save_path = './results/began_64_3e-5/'
        self.mode = 'celeba'
        self.test_size = 0.01
        # nn configs
        self.img_size = 64
        self.latent_size = 64
        self.channels = 3 # num of image channels
        self.img_shape = (self.channels, self.img_size, self.img_size)
        self.num_workers = 4
        self.gamma = 0.4
        self.lambda_k = 0.001
        # training configs
        self.batch_size = 16
        self.random_size = 2*8
        self.n_epochs = 20
        self.lr = 3e-5
        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")