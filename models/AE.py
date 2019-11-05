import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
from progress.bar import Bar

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

from models.base import Discriminator

class AE(object):
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
        self.discriminator = Discriminator(self.img_size).to(self.device)
        self.d_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr)

    def train(self, train_data):
        bar = Bar('Training', max=len(train_data))
        # set to train
        self.discriminator.train()

        running_d = 0

        for step, data in enumerate(train_data):
            # generated sample
            AE_x = self.discriminator(data.to(self.device)) # pass through autoencoder

            d_loss_real = nn.MSELoss()(AE_x, data.to(self.device))

            self.d_optim.zero_grad()
            d_loss = d_loss_real
            d_loss.backward()
            self.d_optim.step()

            running_d += d_loss

            bar.next()
        running_d /= len(train_data)
        bar.finish()
        return running_d

    def test(self, test_data):
        bar = Bar('Testing', max=len(test_data))
        # set to train
        self.discriminator.eval()

        running_c = 0

        for step, data in enumerate(test_data):
            # generated sample
            AE_x = self.discriminator(data.to(self.device))

            d_loss_real = nn.MSELoss()(AE_x, data.to(self.device))
            conv = d_loss_real.item()
            running_c += conv

            bar.next()
        running_c /= len(test_data)
        bar.finish()
        return running_c

    def visualize(self, val_data, savepath):

        # set to eval
        self.discriminator.eval()

        AE_x = self.discriminator(val_data.to(self.device))

        # preprocess generated images
        gen_img = AE_x.cpu().detach().numpy() * 0.5 + 0.5
        gen_img = np.transpose(gen_img * 255.0, (0,2,3,1)).astype(np.uint8)

        # preprocess original images
        ori_img = val_data.cpu().detach().numpy() * 0.5 + 0.5
        ori_img = np.transpose(ori_img * 255.0, (0,2,3,1)).astype(np.uint8)

        # prepare grid on plot
        fig = plt.figure(figsize=(10, 7.5))
        fig.subplots_adjust(left=0, bottom=0, right=1, top=0.95, wspace=None, hspace=None)
        outer = gridspec.GridSpec(2, 1, wspace=0.2, hspace=0.15)

        # plot generated images
        ax = fig.add_subplot(outer[0])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        plt.setp(ax, title='reconstructed images')
        inner = gridspec.GridSpecFromSubplotSpec(2, 8, subplot_spec=outer[0], wspace=0.1, hspace=0.1)
        for i in range(2*8):
            ax = plt.Subplot(fig, inner[i])
            ax.imshow(gen_img[i])
            ax.set_xticks([])
            ax.set_yticks([])
            #plt.setp(ax, title='fake' if AE_fake[i] > 0.5 else 'real')
            fig.add_subplot(ax)

        # plot original images
        ax = fig.add_subplot(outer[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        plt.setp(ax, title='original images')
        inner = gridspec.GridSpecFromSubplotSpec(2, 8, subplot_spec=outer[1], wspace=0.1, hspace=0.1)
        for i in range(2*8):
            ax = plt.Subplot(fig, inner[i])
            ax.imshow(ori_img[i])
            ax.set_xticks([])
            ax.set_yticks([])
            #plt.setp(ax, title='fake' if AE_fake[i] > 0.5 else 'real')
            fig.add_subplot(ax)

        fig.savefig(savepath + 'ae_fig.png', format='png', dpi=300)

    def save_model(self, path):
        self.discriminator.encoder.save_model(path + 'ae_enc')
        self.discriminator.decoder.save_model(path + 'ae_dec')
