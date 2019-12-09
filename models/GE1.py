import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
from progress.bar import Bar

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

from models.base import Generator, ConvAE

class GE1(object):
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
        # to load the weights
        self.weight_dir = config.save_path

    def build_model(self):
        self.generator = Generator(self.img_size).to(self.device)
        self.discriminator = ConvAE(self.img_size).to(self.device)
        # load weights
        self.generator.load_state_dict(torch.load(self.weight_dir + 'gen'))
        self.discriminator.load_state_dict(torch.load(self.weight_dir + 'ae64_ae'))

    def test(self, test_data): # in GE1, we do not have a train function. instead, we only have test
        # set to eval
        self.generator.eval()
        self.discriminator.eval()
        num_to_test = 4

        # prepare grid on plot
        fig = plt.figure(figsize=(10, num_to_test*7.5))
        fig.subplots_adjust(left=0, bottom=0, right=1, top=0.95, wspace=None, hspace=None)
        outer = gridspec.GridSpec(3*num_to_test, 1, wspace=0.2, hspace=0.15)

        for step, data in enumerate(test_data):
            # initialize noise vectors
            z = Variable(torch.rand(self.batch_size, self.latent_size).to(self.device).mul(2).add(-1), requires_grad=True)
            #z = Variable(torch.randn(self.batch_size, self.latent_size).to(self.device), requires_grad=True)
            # initialize optimizer
            z_optim = torch.optim.Adam([z], lr=0.1)

            num_iterations = 500
            bar = Bar('Step ' + str(step), max=num_iterations)
            for i in range(num_iterations):
                z_optim.zero_grad()
                m = self.discriminator.encode(data.to(self.device))
                ge = self.discriminator.encode(self.generator(z))

                loss = nn.MSELoss()(ge, m) + 0.01*torch.norm(z)
                loss.backward()
                z_optim.step()
                if (i+1) % 100 == 0:
                    print(" loss: {:.4f}".format(loss))
                    # after optimization, we get result by updates z
                    gen_img = self.generator(z).cpu().detach().numpy() * 0.5 + 0.5
                    gen_img = np.transpose(gen_img, (0,2,3,1))
                    real_img = data.cpu().detach().numpy() * 0.5 + 0.5
                    real_img = np.transpose(real_img, (0,2,3,1))
                    just_ae_img = self.discriminator(data.to(self.device)).cpu().detach().numpy() * 0.5 + 0.5
                    just_ae_img = np.transpose(just_ae_img, (0,2,3,1))

                    # plot generated images
                    ax = fig.add_subplot(outer[3*step])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.axis('off')
                    plt.setp(ax, title='reconstructed images')
                    inner = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=outer[3*step], wspace=0.1, hspace=0.1)
                    for i in range(1*8):
                        ax = plt.Subplot(fig, inner[i])
                        ax.imshow(gen_img[i])
                        ax.set_xticks([])
                        ax.set_yticks([])
                        #plt.setp(ax, title='fake' if AE_fake[i] > 0.5 else 'real')
                        fig.add_subplot(ax)

                    # plot ae reconstructed images
                    ax = fig.add_subplot(outer[3*step+1])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.axis('off')
                    plt.setp(ax, title='ae images')
                    inner = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=outer[3*step+1], wspace=0.1, hspace=0.1)
                    for i in range(1*8):
                        ax = plt.Subplot(fig, inner[i])
                        ax.imshow(just_ae_img[i])
                        ax.set_xticks([])
                        ax.set_yticks([])
                        #plt.setp(ax, title='fake' if AE_fake[i] > 0.5 else 'real')
                        fig.add_subplot(ax)

                    # plot original images
                    ax = fig.add_subplot(outer[3*step+2])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.axis('off')
                    plt.setp(ax, title='original images')
                    inner = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=outer[3*step+2], wspace=0.1, hspace=0.1)
                    for i in range(1*8):
                        ax = plt.Subplot(fig, inner[i])
                        ax.imshow(real_img[i])
                        ax.set_xticks([])
                        ax.set_yticks([])
                        #plt.setp(ax, title='fake' if AE_fake[i] > 0.5 else 'real')
                        fig.add_subplot(ax)
                    if step == (num_to_test - 1):
                        break
                    fig.savefig(self.weight_dir + 'ge1_fig.png', format='png', dpi=300)

                bar.next()
            bar.finish()
            print("loss = {:.4f}".format(loss))

            # after optimization, we get result by updates z
            gen_img = self.generator(z).cpu().detach().numpy() * 0.5 + 0.5
            gen_img = np.transpose(gen_img, (0,2,3,1))
            real_img = data.cpu().detach().numpy() * 0.5 + 0.5
            real_img = np.transpose(real_img, (0,2,3,1))
            just_ae_img = self.discriminator(data.to(self.device)).cpu().detach().numpy() * 0.5 + 0.5
            just_ae_img = np.transpose(just_ae_img, (0,2,3,1))

            # plot generated images
            ax = fig.add_subplot(outer[3*step])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
            plt.setp(ax, title='reconstructed images')
            inner = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=outer[3*step], wspace=0.1, hspace=0.1)
            for i in range(1*8):
                ax = plt.Subplot(fig, inner[i])
                ax.imshow(gen_img[i])
                ax.set_xticks([])
                ax.set_yticks([])
                #plt.setp(ax, title='fake' if AE_fake[i] > 0.5 else 'real')
                fig.add_subplot(ax)

            # plot ae reconstructed images
            ax = fig.add_subplot(outer[3*step+1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
            plt.setp(ax, title='ae images')
            inner = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=outer[3*step+1], wspace=0.1, hspace=0.1)
            for i in range(1*8):
                ax = plt.Subplot(fig, inner[i])
                ax.imshow(just_ae_img[i])
                ax.set_xticks([])
                ax.set_yticks([])
                #plt.setp(ax, title='fake' if AE_fake[i] > 0.5 else 'real')
                fig.add_subplot(ax)

            # plot original images
            ax = fig.add_subplot(outer[3*step+2])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
            plt.setp(ax, title='original images')
            inner = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=outer[3*step+2], wspace=0.1, hspace=0.1)
            for i in range(1*8):
                ax = plt.Subplot(fig, inner[i])
                ax.imshow(real_img[i])
                ax.set_xticks([])
                ax.set_yticks([])
                #plt.setp(ax, title='fake' if AE_fake[i] > 0.5 else 'real')
                fig.add_subplot(ax)
            if step == (num_to_test - 1):
                break
            fig.savefig(self.weight_dir + 'ge1_fig.png', format='png', dpi=300)

        fig.savefig(self.weight_dir + 'ge1_fig.png', format='png', dpi=300)