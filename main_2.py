import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")

from config import Config
from data_loader import make_loader
from models.BEGAN import BEGAN

def get_folder_dir(mode):
    if mode == 'celeba':
        return 'CelebA', 'img_align_celeba'

# create config
config = Config()

# create dataloader
train, test = make_loader(
    config.dataset_path,
    config.batch_size,
    config.mode,
    config.img_size,
    config.num_workers
)

for step, data in enumerate(test):
    val_data = data[0:config.random_size]
    break

# create model
began = BEGAN(config, val_data)
began.build_model()
began.load_model()

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

print("model structure")
print_network(began.generator)
print_network(began.discriminator)
print()

began.train_ae(train, config.n_epochs)