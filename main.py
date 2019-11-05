import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
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

z_G = Variable(torch.FloatTensor(config.random_size, config.latent_size)).to(config.device)
z_G.data.normal_(0,1)

conv_scores = []
best_c = 1e10
for epoch in range(config.n_epochs):
    running_g, running_d = began.train(train)
    running_c = began.test(test)
    conv_scores.append(running_c)
    print("Epoch {} - G-loss = {:.4f}, D-loss = {:.4f}".format(epoch+1, running_g, running_d))
    print("Convergence value {:.4f}".format(running_c))
    np.save(config.save_path + 'conv_scores', conv_scores)
    if running_c < best_c:
        # model improved, so save
        began.save_model(config.save_path)
        best_c = running_c
    if epoch + 1 % 10 == 0:
        # save checkpoint
        began.visualize(z_G, config.save_path + 'ep' + str(epoch+1) + '_')