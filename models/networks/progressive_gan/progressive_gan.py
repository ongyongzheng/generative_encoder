# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.optim as optim

from ..gan.base_GAN import BaseGAN
from ...utils.config import BaseConfig
from .networks.progressive_conv_net import GNet, DNet


class ProgressiveGAN(BaseGAN):
    """
    Implementation of NVIDIA's progressive GAN.
    """

    def __init__(self,
                 config=None,
                 **kwargs):
        """
        Args:

        Specific Arguments:
            - depthScale0 (int)
            - initBiasToZero (bool): should layer's bias be initialized to
                                     zero ?
            - leakyness (float): negative slope of the leakyRelU activation
                                 function
            - perChannelNormalization (bool): do we normalize the output of
                                              each convolutional layer ?
            - miniBatchStdDev (bool): mini batch regularization for the
                                      discriminator
            - equalizedlR (bool): if True, forces the optimizer to see weights
                                  in range (-1, 1)

        """
        if not 'config' in vars(self):
            self.config = BaseConfig()

        self.config.depthScale0 = config["depthScale0"]
        self.config.initBiasToZero = config["initBiasToZero"]
        self.config.leakyReluLeak = config["leakyness"]
        self.config.depthOtherScales = []
        self.config.perChannelNormalization = config["perChannelNormalization"]
        self.config.alpha = 0
        self.config.miniBatchStdDev = config["miniBatchStdDev"]
        self.config.equalizedlR = config["equalizedlR"]
        self.config.miniBatchNormalization = config["miniBatchNormalization"]
        self.config.normalization = config["normalization"]

        BaseGAN.__init__(self, config=config, **kwargs)

    def getNetG(self):

        gnet = GNet(self.config,
                    generationActivation=self.lossCriterion.generationActivation)

        # Add scales if necessary
        for depth in self.config.depthOtherScales:
            gnet.addScale(depth)

        # If new scales are added, give the generator a blending layer
        if self.config.depthOtherScales:
            gnet.setNewAlpha(self.config.alpha)

        return gnet

    def getNetD(self):

        dnet = DNet(self.config)

        # Add scales if necessary
        for depth in self.config.depthOtherScales:
            dnet.addScale(depth)

        # If new scales are added, give the discriminator a blending layer
        if self.config.depthOtherScales:
            dnet.setNewAlpha(self.config.alpha)

        return dnet

    def getOptimizerD(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.netD.parameters()),
                          betas=[0, 0.99], lr=self.config.learningRate)

    def getOptimizerG(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()),
                          betas=[0, 0.99], lr=self.config.learningRate)

    def addScale(self, depthNewScale):
        """
        Add a new scale to the model. The output resolution becomes twice
        bigger.
        """
        self.netG = self.getOriginalG()
        self.netD = self.getOriginalD()

        self.netG.addScale(depthNewScale)
        self.netD.addScale(depthNewScale)

        self.config.depthOtherScales.append(depthNewScale)

        self.updateSolversDevice()

    def updateAlpha(self, newAlpha):
        """
        Update the blending factor alpha.

        Args:
            - alpha (float): blending factor (in [0,1]). 0 means only the
                             highest resolution in considered (no blend), 1
                             means the highest resolution is fully discarded.
        """
        print("Changing alpha to %.3f" % newAlpha)

        self.getOriginalG().setNewAlpha(newAlpha)
        self.getOriginalD().setNewAlpha(newAlpha)

        if self.avgG:
            self.avgG.module.setNewAlpha(newAlpha)

        self.config.alpha = newAlpha

    def getSize(self):
        """
        Get output image size (W, H)
        """
        return self.getOriginalG().getOutputSize()
