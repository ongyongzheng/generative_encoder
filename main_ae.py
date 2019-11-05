import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import os

from config import Config
from data_loader import make_loader
from models.AE import AE

def get_folder_dir(mode):
    if mode == 'celeba':
        return 'CelebA', 'img_align_celeba'

# create config
config = Config()

# create model
ae = AE(config)
ae.build_model()

print("model structure")
print(ae.discriminator)
print()

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

conv_scores = []
best_c = 1e10
for epoch in range(config.n_epochs):
    running_d = ae.train(train)
    running_c = ae.test(test)
    conv_scores.append(running_c)
    print("Epoch {} - AE-loss = {:.4f}".format(epoch+1, running_d))
    print("Test Loss value {:.4f}".format(running_c))
    np.save(config.save_path + 'ae_conv_scores', conv_scores)
    if running_c < best_c:
        # model improved, so save
        ae.save_model(config.save_path)
        best_c = running_c
    if (epoch + 1) % 10 == 0:
        # save checkpoint
        print("saving image")
        ae.visualize(val_data, config.save_path + 'ep' + str(epoch+1) + '_')