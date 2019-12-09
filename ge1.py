import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import os

from config import Config
from data_loader import make_loader
from models.GE1 import GE1

def get_folder_dir(mode):
    if mode == 'celeba':
        return 'CelebA', 'img_align_celeba'

# create config
config = Config()

# create model
ge1 = GE1(config)
ge1.build_model()

print("model structure")
print(ge1.generator)
print(ge1.discriminator)
print()

# create dataloader
train, test = make_loader(
    config.dataset_path,
    config.batch_size,
    config.mode,
    config.img_size,
    config.num_workers
)

ge1.test(test)