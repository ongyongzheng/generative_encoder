import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
from progress.bar import Bar

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

from models.base import Generator, Discriminator

class BEGAN(object):
    def __init__(self, config, val_data):
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
        self.global_epoch = 0
        self.weight_dir = config.save_path
        self.fixed = torch.rand(2*8, self.latent_size).to(self.device).mul(2).add(-1)
        #self.fixed = torch.randn(8*8, self.latent_size).to(self.device)
        self.k_t = 0
        self.val_data = val_data

    def weight_init(self, mean, std):
        self.generator.weight_init(mean, std)
        self.discriminator.weight_init(mean, std)

    def load_model(self):
        # load weights
        self.generator.load_state_dict(torch.load(self.weight_dir + 'gen'))
        #self.discriminator.encoder.load_state_dict(torch.load(self.weight_dir + 'enc'))
        #self.discriminator.decoder.load_state_dict(torch.load(self.weight_dir + 'dec'))

    def build_model(self):
        self.generator = Generator(self.img_size).to(self.device)
        self.discriminator = Discriminator(self.img_size).to(self.device)

        self.g_optim = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.lr,
            betas=(0.5, 0.999))
        self.d_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr,
            betas=(0.5, 0.999))
        self.d_optim_scheduler = lr_scheduler.StepLR(self.d_optim, step_size=1, gamma=0.5)
        self.g_optim_scheduler = lr_scheduler.StepLR(self.g_optim, step_size=1, gamma=0.5)

        self.weight_init(0.0, 0.02)

    def scheduler_step(self):
        self.d_optim_scheduler.step()
        self.g_optim_scheduler.step()

    def train(self, train_data, num_epochs=20):
        conv_scores = []
        gen_loss = []
        dis_loss = []
        d_real = []
        d_fake = []
        best_c = 1e10
        lr_step_size = len(train_data.dataset)//self.batch_size*self.n_epochs//8
        print("LR Step Size: {}".format(lr_step_size))
        self.k_t = 0
        self.global_epoch = 0
        self.global_iter = 0

        for epoch in range(num_epochs):
            self.global_epoch += 1
            for step, data in enumerate(train_data):
                self.global_iter += 1
                # set to train
                self.generator.train()
                self.discriminator.train()

                # train discriminator
                AE_x = self.discriminator(data.to(self.device))
                z_G = torch.rand(self.batch_size, self.latent_size).to(self.device).mul(2).add(-1)
                #z_G = torch.randn(self.batch_size, self.latent_size).to(self.device)
                sample_z_G = self.generator(z_G)
                AE_fake = self.discriminator(sample_z_G.detach())

                d_loss_real = nn.L1Loss()(AE_x, data.to(self.device))
                d_loss_fake = nn.L1Loss()(AE_fake, sample_z_G)

                d_loss = d_loss_real - self.k_t * d_loss_fake
                self.d_optim.zero_grad()
                d_loss.backward()
                self.d_optim.step()

                # train generator
                z_G = torch.rand(self.batch_size, self.latent_size).to(self.device).mul(2).add(-1)
                #z_G = torch.randn(self.batch_size, self.latent_size).to(self.device)
                sample_z_G = self.generator(z_G)
                AE_fake = self.discriminator(sample_z_G)

                g_loss = nn.L1Loss()(sample_z_G, AE_fake)
                self.g_optim.zero_grad()
                g_loss.backward()
                self.g_optim.step()

                # update kt
                g_d_balance = (self.gamma * d_loss_real - g_loss).item()
                self.k_t = max(min(self.k_t + self.lambda_k*g_d_balance, 1.0), 0.0)

                # visualization
                if self.global_iter % 500 == 0:
                    m_global = (d_loss_real.data + abs(g_d_balance)).cpu()
                    print()
                    print('Iter:{}, M:{:.3f}, k_t:{:.3f}'.format(self.global_iter,m_global.item(),self.k_t))
                    print('d_loss_real:{:.3f}, d_loss_fake:{:.3f}, g_loss:{:.3f}'.format(
                        d_loss_real.cpu().item(),
                        d_loss_fake.cpu().item(),
                        g_loss.cpu().item()
                    ))
                    # save image
                    self.visualize(self.weight_dir + 'sub_ep' + str(epoch+1) + '_' + str(self.global_iter) + '_')
                    self.visualize_ae(self.val_data, self.weight_dir)
                    conv_scores.append(m_global.item())
                    gen_loss.append(g_loss.cpu().item())
                    d_real.append(d_loss_real.cpu().item())
                    d_fake.append(d_loss_fake.cpu().item())
                    np.save(self.weight_dir + 'conv_scores', np.array(conv_scores))
                    np.save(self.weight_dir + 'gen_loss', np.array(gen_loss))
                    np.save(self.weight_dir + 'd_real', np.array(d_real))
                    np.save(self.weight_dir + 'd_fake', np.array(d_fake))
                    plt.close()
                    plt.plot(np.array(gen_loss), 'g-', label='g_loss')
                    plt.plot(np.array(d_real), 'b-', label='d_real')
                    plt.plot(np.array(d_fake), 'r-', label='d_fake')
                    plt.legend(loc="upper right")
                    plt.savefig(self.weight_dir + 'graph.png', format='png', dpi=300)
                    plt.close()
                    plt.plot(np.array(conv_scores))
                    plt.savefig(self.weight_dir + 'conv.png', format='png', dpi=300)
                    plt.close()

                    if m_global < best_c:
                        self.save_model(self.weight_dir)

                if self.global_iter % lr_step_size == 0:
                    self.scheduler_step()

            self.visualize(self.weight_dir, use_fixed=True)

    def train_ae(self, train_data, num_epochs=20):
        d_real = []
        d_fake = []
        lr_step_size = len(train_data.dataset)//self.batch_size*self.n_epochs//8
        print("LR Step Size: {}".format(lr_step_size))
        self.global_epoch = 0
        self.global_iter = 0
        for epoch in range(num_epochs):
            self.global_epoch += 1
            for step, data in enumerate(train_data):
                self.global_iter += 1
                # set to train
                self.generator.eval()
                self.discriminator.train()

                # train discriminator
                AE_x = self.discriminator(data.to(self.device))
                z_G = torch.rand(self.batch_size, self.latent_size).to(self.device).mul(2).add(-1)
                #z_G = torch.randn(self.batch_size, self.latent_size).to(self.device)
                sample_z_G = self.generator(z_G)
                AE_fake = self.discriminator(sample_z_G.detach())

                d_loss_real = nn.L1Loss()(AE_x, data.to(self.device))
                d_loss_fake = nn.L1Loss()(AE_fake, sample_z_G)

                d_loss = d_loss_real + d_loss_fake
                self.d_optim.zero_grad()
                d_loss.backward()
                self.d_optim.step()

                # visualization
                if self.global_iter % 500 == 0:
                    print()
                    print('Iter:{}, Loss:{:.3f}'.format(self.global_iter,d_loss))
                    print('d_loss_real:{:.3f}, d_loss_fake:{:.3f}'.format(
                        d_loss_real.cpu().item(),
                        d_loss_fake.cpu().item()
                    ))
                    # save image
                    #self.visualize(self.weight_dir + 'sub_ae_ep' + str(epoch+1) + '_' + str(self.global_iter) + '_')
                    d_real.append(d_loss_real.cpu().item())
                    d_fake.append(d_loss_fake.cpu().item())
                    np.save(self.weight_dir + 'ae_d_real', np.array(d_real))
                    np.save(self.weight_dir + 'ae_d_fake', np.array(d_fake))
                    plt.close()
                    plt.plot(np.array(d_real), 'b-', label='d_real')
                    plt.plot(np.array(d_fake), 'r-', label='d_fake')
                    plt.legend(loc="upper right")
                    plt.savefig(self.weight_dir + 'ae_graph.png', format='png', dpi=300)
                    plt.close()
                    self.visualize_ae(self.val_data, self.weight_dir + 'ae_t_')

                    self.save_model(self.weight_dir + 'ae_')

                if self.global_iter % lr_step_size == 0:
                    self.scheduler_step()

            #self.visualize(self.weight_dir + 'ae_ep' + str(epoch+1) + '_', use_fixed=True)

    def visualize(self, savepath, use_fixed=True):

        # set to eval
        self.generator.eval()
        self.discriminator.eval()

        z_G = torch.rand(2*8, self.latent_size).to(self.device).mul(2).add(-1)
        #z_G = torch.randn(8*8, self.latent_size).to(self.device)
        if use_fixed:
            z_G = self.fixed

        sample_z_G = self.generator(z_G)
        AE_fake = self.discriminator(sample_z_G)

        # preprocess generated images
        gen_img = sample_z_G.cpu().detach().numpy() * 0.5 + 0.5
        gen_img = np.transpose(gen_img, (0,2,3,1))#.astype(np.uint8)

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
        plt.close()

    def visualize_ae(self, val_data, savepath):

        # set to eval
        self.discriminator.eval()

        AE_x = self.discriminator(val_data.to(self.device))

        # preprocess generated images
        gen_img = AE_x.cpu().detach().numpy() * 0.5 + 0.5
        gen_img = np.transpose(gen_img, (0,2,3,1))#.astype(np.uint8)

        # preprocess original images
        ori_img = val_data.cpu().detach().numpy() * 0.5 + 0.5
        ori_img = np.transpose(ori_img, (0,2,3,1))#.astype(np.uint8)

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
        self.generator.save_model(path + 'gen')
        self.discriminator.encoder.save_model(path + 'enc')
        self.discriminator.decoder.save_model(path + 'dec')
