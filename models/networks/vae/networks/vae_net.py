from collections import OrderedDict
import math
import numpy as np

import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        y = m.in_features
        m.weight.data.normal_(0.0, 1/np.sqrt(y))
        m.bias.data.fill_(0)

def var_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.fill_(0)
        m.bias.data.fill_(0)

class ENet(nn.Module):
    def __init__(self,
                 config):
        """
        Parameters:
            self.config.dimOutput,
            self.config.dimE,
            self.config.latentVectorDim,
            imageSize=self.config.imageSize
        """
        super(ENet, self).__init__()

        self.imageSize = config.imageSize
        self.dataType = config.dataType
        if (self.imageSize & (self.imageSize - 1)) != 0:
            # check power of 2
            raise ValueError("self.imageSize should be a power of 2!")
        depthModel = int(math.log2(self.imageSize)) - 1

        # build hidden_dims
        self.hidden_dims = [config.dimE]
        for _ in range(depthModel - 1):
            self.hidden_dims.append(self.hidden_dims[-1] * 2)

        currDepth = config.dimOutput # start with input dimension

        sequence = OrderedDict([])

        # build the conv-nn based on the imageSize.
        # the final layer size is [-1, hidden_dims[-1], 2, 2]
        for index, h_dim in enumerate(self.hidden_dims):
            sequence["conv" +
                     str(index)] = nn.Conv2d(currDepth, h_dim,
                                             4, 2, 1, bias=False)
            sequence["batchNorm" + str(index)] = nn.BatchNorm2d(h_dim)
            sequence["relu" + str(index)] = nn.LeakyReLU(0.2, inplace=True)

            currDepth = h_dim

        self.dimFeatureMap = currDepth

        self.main = nn.Sequential(sequence)
        self.main.apply(weights_init)

        self.initMuLayer(config.latentVectorDim)
        self.initVarLayer(config.latentVectorDim)

    def initMuLayer(self, sizeDecisionLayer):
        if self.dataType == "stft_signal":
            self.muLayer = nn.Linear(
                self.dimFeatureMap * 2 * 2, sizeDecisionLayer)
            self.muLayer.apply(weights_init)
            self.sizeDecisionLayer = sizeDecisionLayer
        else:
            self.muLayer = nn.Linear(
                self.dimFeatureMap * 2 * 2, sizeDecisionLayer)
            self.muLayer.apply(weights_init)
            self.sizeDecisionLayer = sizeDecisionLayer

    def initVarLayer(self, sizeDecisionLayer):
        if self.dataType == "stft_signal":
            self.varLayer = nn.Linear(
                self.dimFeatureMap * 2 * 2, sizeDecisionLayer)
            self.varLayer.apply(var_init) # apply zero initialization to variance layer for stability
            self.sizeDecisionLayer = sizeDecisionLayer
        else:
            self.varLayer = nn.Linear(
                self.dimFeatureMap * 2 * 2, sizeDecisionLayer)
            self.varLayer.apply(var_init) # apply zero initialization to variance layer for stability
            self.sizeDecisionLayer = sizeDecisionLayer

    def reparameterize(self, mu, logvar):
        """
        Code to sample from N(mu, var) from N(0,1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        x = self.main(input)

        # convert to vector
        x = torch.flatten(x, start_dim=1)

        x_mu = self.muLayer(x)
        x_var = self.varLayer(x)
        z = self.reparameterize(x_mu, x_var)

        return z, x_mu, x_var


class DNet(nn.Module):
    def __init__(self,
                 config,
                 generationActivation=None):
        """
        Parameters:
            self.config.latentVectorDim,
            self.config.dimOutput,
            self.config.dimD,
            imageSize=self.config.imageSize,
            dataType=self.config.dataType
        """
        super(DNet, self).__init__()

        self.imageSize = config.imageSize
        self.dataType = config.dataType
        if (self.imageSize & (self.imageSize - 1)) != 0:
            # check power of 2
            raise ValueError("self.imageSize should be a power of 2!")
        depthModel = int(math.log2(self.imageSize)) - 1

        # build hidden_dims
        self.hidden_dims = [config.dimD]
        for _ in range(depthModel - 1):
            self.hidden_dims.append(self.hidden_dims[-1] * 2)
        self.hidden_dims.reverse()

        self.initFormatLayer(config.latentVectorDim)

        sequence = OrderedDict([])

        # build the conv-nn based on the imageSize.
        # the final layer size is [-1, dimOutput, imageSize, imageSize]
        for index in range(len(self.hidden_dims) - 1):
            sequence["convTranspose" +
                     str(index)] = nn.ConvTranspose2d(
                        self.hidden_dims[index], self.hidden_dims[index + 1], 4, 2, 1, bias=False)
            sequence["batchNorm" + str(index)] = nn.BatchNorm2d(self.hidden_dims[index + 1])
            sequence["relu" + str(index)] = nn.LeakyReLU(0.2, inplace=True)

        if self.dataType in ["image", "1d_signal"]:
            sequence["outlayer"] = nn.ConvTranspose2d(
                config.dimD, config.dimOutput, 4, 2, 1, bias=False)

            self.outputAcctivation = generationActivation
        
        elif self.dataType in ["stft_signal"]:
            sequence["outlayer"] = nn.ConvTranspose2d(
                config.dimD, 2, 4, 2, 1, bias=False) # in stft, output is of channel 2

            self.outputAcctivation = generationActivation

        else:
            raise NotImplementedError("given datatype {} is not implemented yet!".format(self.dataType))

        self.main = nn.Sequential(sequence)
        self.main.apply(weights_init)

    def initFormatLayer(self, dimLatentVector):

        if self.dataType == "stft_signal":
            self.formatLayer = nn.Linear(
                dimLatentVector, self.hidden_dims[0] * 2 * 2)
            self.formatLayer.apply(weights_init)
        else:
            self.formatLayer = nn.Linear(
                dimLatentVector, self.hidden_dims[0] * 2 * 2)
            self.formatLayer.apply(weights_init)

    def forward(self, input):

        x = self.formatLayer(input)
        if self.dataType == "stft_signal":
            x = x.view(-1, self.hidden_dims[0], 2, 2)
        else:
            x = x.view(-1, self.hidden_dims[0], 2, 2)
        x = self.main(x)

        if self.outputAcctivation is None:
            return x

        return self.outputAcctivation(x)
