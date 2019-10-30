import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os

from config import Config
from data_loader import make_loader
from models.BEGAN import BEGAN

def get_folder_dir(mode):
    if mode == 'celeba':
        return 'CelebA', 'img_align_celeba'

# create config
config = Config()

# create model
began = BEGAN(config)
began.build_model()

print("model structure")
print(began.generator)
print(began.discriminator)
print()

# create dataloader
train, test = make_loader(
    config.dataset_path,
    config.batch_size,
    config.mode,
    config.img_size,
    config.num_workers
)

began.train(train)