import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
from progress.bar import Bar

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

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
        self.gamma = config.gamma
        self.lambda_k = config.lambda_k

    def build_model(self):
        self.generator = Generator(self.img_size, self.latent_size, self.channels).to(self.device)
        self.discriminator = Discriminator(self.img_size, self.latent_size, self.channels).to(self.device)

        self.g_optim = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.lr)
        self.d_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr)

    def train(self, train_data):
        bar = Bar('Training', max=len(train_data))
        # set to train
        self.generator.train()
        self.discriminator.train()

        z_D = Variable(torch.FloatTensor(self.batch_size, self.latent_size)).to(self.device)
        z_G = Variable(torch.FloatTensor(self.batch_size, self.latent_size)).to(self.device)
        #z_fixed = Variable(torch.FloatTensor(self.batch_size, self.latent_size).normal_(0, 1)).to(self.device)

        k_t = 0

        running_g = 0
        running_d = 0

        for step, data in enumerate(train_data):
            # initialize noise vectors
            z_D.data.normal_(0,1)
            z_G.data.normal_(0,1)

            # generated sample
            sample_z_G = self.generator(z_G)
            AE_x = self.discriminator(data.to(self.device))
            AE_fake = self.discriminator(sample_z_G)

            d_loss_real = nn.L1Loss()(AE_x, data.to(self.device))
            d_loss_fake = nn.L1Loss()(AE_fake, sample_z_G)

            self.d_optim.zero_grad()
            d_loss = d_loss_real - k_t * d_loss_fake
            d_loss.backward()
            self.d_optim.step()

            sample_z_G = self.generator(z_G)
            AE_fake = self.discriminator(sample_z_G)

            self.g_optim.zero_grad()
            g_loss = nn.L1Loss()(sample_z_G, AE_fake)
            g_loss.backward()
            self.g_optim.step()

            running_g += g_loss
            running_d += d_loss

            g_d_balance = (self.gamma * d_loss_real - d_loss_fake).item()
            k_t += self.lambda_k * g_d_balance
            k_t = max(min(1, k_t), 0)

            bar.next()
        running_g /= len(train_data)
        running_d /= len(train_data)
        bar.finish()
        return running_g, running_d


    def test(self, test_data):
        bar = Bar('Testing', max=len(test_data))
        # set to train
        self.generator.eval()
        self.discriminator.eval()

        z_D = Variable(torch.FloatTensor(self.batch_size, self.latent_size)).to(self.device)
        z_G = Variable(torch.FloatTensor(self.batch_size, self.latent_size)).to(self.device)

        running_c = 0

        for step, data in enumerate(test_data):
            # initialize noise vectors
            z_D.data.normal_(0,1)
            z_G.data.normal_(0,1)

            # generated sample
            sample_z_G = self.generator(z_G)
            AE_x = self.discriminator(data.to(self.device))
            AE_fake = self.discriminator(sample_z_G)

            d_loss_real = nn.L1Loss()(AE_x, data.to(self.device))
            d_loss_fake = nn.L1Loss()(AE_fake, sample_z_G)

            conv = d_loss_real.item() + np.absolute((self.gamma * d_loss_real - d_loss_fake).item())

            running_c += conv

            bar.next()
        running_c /= len(test_data)
        bar.finish()
        return running_c

    def visualize(self, z_G, savepath):

        # set to eval
        self.generator.eval()
        self.discriminator.eval()

        sample_z_G = self.generator(z_G)
        AE_fake = self.discriminator(sample_z_G)

        # preprocess generated images
        gen_img = sample_z_G.detach().numpy() * 0.5 + 0.5
        gen_img = np.transpose(gen_img * 255.0, (0,2,3,1)).astype(np.uint8)

        # prepare grid on plot
        fig = plt.figure(figsize=(10, 7.5))
        fig.subplots_adjust(left=0, bottom=0, right=1, top=0.95, wspace=None, hspace=None)
        outer = gridspec.GridSpec(1, 1, wspace=0.2, hspace=0.15)

        # plot generated images
        ax = fig.add_subplot(outer[0])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        plt.setp(ax, title='generated images')
        inner = gridspec.GridSpecFromSubplotSpec(2, 8, subplot_spec=outer[0], wspace=0.1, hspace=0.1)
        for i in range(2*8):
            ax = plt.Subplot(fig, inner[i])
            ax.imshow(gen_img[i])
            ax.set_xticks([])
            ax.set_yticks([])
            #plt.setp(ax, title='fake' if AE_fake[i] > 0.5 else 'real')
            fig.add_subplot(ax)

        fig.savefig(savepath + 'fig.png', format='png', dpi=300)

    def save_model(self, path):
        self.generator.save_model(path + 'gen')
        self.discriminator.encoder.save_model(path + 'enc')
        self.discriminator.decoder.save_model(path + 'dec')
