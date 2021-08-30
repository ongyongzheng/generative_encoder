# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from copy import deepcopy

import torch
import torch.nn as nn

from ..base.base_NET import BaseNET
from ...utils.config import BaseConfig, updateConfig
from .loss_criterions import base_loss_criterions
from .loss_criterions.gradient_losses import WGANGPGradientPenalty
from ...utils.utils import loadPartOfStateDict, finiteCheck, \
    loadStateDictCompatible


class BaseGAN(BaseNET):
    """Abstract class: the basic framework for GAN training.
    """

    def __init__(self,
                 config=None,
                 **kwargs):
        """
        Initialize the BaseGAN

        In addition to base parameters, GAN additionally requires
        1. dimLatentVector
        2. lossMode
        """

        BaseNET.__init__(self, config=config, **kwargs)

        self.valid_losses = ['WGANGP']

        if config["lossMode"] not in self.valid_losses:
            raise ValueError(
                "lossMode should be one of the following : {}".format(self.valid_losses))

        # Latent vector dimension
        self.config.noiseVectorDim = config["dimLatentVector"]
        self.config.latentVectorDim = self.config.noiseVectorDim

        # Loss criterion
        self.config.lossCriterion = config["lossMode"]
        self.lossCriterion = getattr(
            base_loss_criterions, config["lossMode"])(self.device)

        # WGAN-GP
        self.config.lambdaGP = config["lambdaGP"]

        # Weight on D's output
        self.config.epsilonD = config["epsilonD"]

        # Initialize the generator and the discriminator
        self.netD = self.getNetD()
        self.netG = self.getNetG()

        # Move the networks to the gpu
        self.updateSolversDevice()


    def test(self, input, getAvG=False, toCPU=True):
        """
        Generate some data given the input latent vector.

        Args:
            input (torch.tensor): input latent vector
        """
        input = input.to(self.device)
        if getAvG:
            if toCPU:
                return self.avgG(input).cpu()
            else:
                return self.avgG(input)
        elif toCPU:
            return self.netG(input).detach().cpu()
        else:
            return self.netG(input).detach()

    def buildAvG(self):
        """
        Create and upload a moving average generator.
        """
        self.avgG = deepcopy(self.getOriginalG())
        for param in self.avgG.parameters():
            param.requires_grad = False

        if self.useGPU:
            self.avgG = nn.DataParallel(self.avgG)
            self.avgG.to(self.device)

    def optimizeParameters(self, input_batch):
        """
        perform a single optimization step

        Args:
            input (torch.tensor): input batch of real data

        """

        allLosses = {}

        # Retrieve the input data
        self.real_input = input_batch.to(self.device)

        n_samples = self.real_input.size()[0]

        # A: Update the discriminator
        self.optimizerD.zero_grad()

        # #1 Real data
        predRealD = self.netD(self.real_input, False)

        lossD = self.lossCriterion.getCriterion(predRealD, True)
        allLosses["lossD_real"] = lossD.item()

        # #2 Fake data
        inputLatent = self.buildNoiseData(n_samples)
        predFakeG = self.netG(inputLatent).detach()
        predFakeD = self.netD(predFakeG, False) # TODO: investigate the False

        lossDFake = self.lossCriterion.getCriterion(predFakeD, False)
        allLosses["lossD_fake"] = lossDFake.item()
        lossD += lossDFake

        # #3 WGANGP gradient loss
        if self.config.lambdaGP > 0:
            allLosses["lossD_Grad"] = WGANGPGradientPenalty(self.real_input,
                                                            predFakeG,
                                                            self.netD,
                                                            self.config.lambdaGP,
                                                            backward=True)

        lossD.backward(retain_graph=True)
        # finiteCheck(self.getOriginalD().parameters()) # TODO: investigate this
        self.optimizerD.step()

        # #4 Epsilon loss
        if self.config.epsilonD > 0:
            lossEpsilon = (predRealD[:, 0] ** 2).sum() * self.config.epsilonD
            lossD += lossEpsilon
            allLosses["lossD_Epsilon"] = lossEpsilon.item()

        # Logs
        lossD = 0
        for key, val in allLosses.items():

            if key.find("lossD") == 0:
                lossD += val

        allLosses["lossD"] = lossD

        # B: Update the generator
        self.optimizerG.zero_grad()
        self.optimizerD.zero_grad()

        # #1 Image generation
        inputNoise = self.buildNoiseData(n_samples)
        predFakeG = self.netG(inputNoise)

        # #2 Status evaluation
        predFakeD, phiGFake = self.netD(predFakeG, True) # TODO: investigate the True

        lossGFake = self.lossCriterion.getCriterion(predFakeD, True)
        allLosses["lossG_fake"] = lossGFake.item()

        lossGFake.backward(retain_graph=True)
        # finiteCheck(self.getOriginalG().parameters()) # TODO: investigate this
        self.optimizerG.step()

        # Logs
        lossG = 0
        for key, val in allLosses.items():

            if key.find("lossG") == 0:
                lossG += val

        allLosses["lossG"] = lossG

        # Update the moving average if relevant
        for p, avg_p in zip(self.getOriginalG().parameters(),
                            self.getOriginalAvgG().parameters()):
            avg_p.mul_(0.999).add_(0.001, p.data)

        return allLosses

    def updateSolversDevice(self, buildAvG=True):
        """
        Move the current networks and solvers to the GPU.
        This function must be called each time netG or netD is modified
        """
        if buildAvG:
            self.buildAvG()

        if not isinstance(self.netD, nn.DataParallel) and self.useGPU:
            self.netD = nn.DataParallel(self.netD)
        if not isinstance(self.netG, nn.DataParallel) and self.useGPU:
            self.netG = nn.DataParallel(self.netG)

        self.netD.to(self.device)
        self.netG.to(self.device)

        self.optimizerD = self.getOptimizerD()
        self.optimizerG = self.getOptimizerG()

        self.optimizerD.zero_grad()
        self.optimizerG.zero_grad()

    def buildNoiseData(self, n_samples):
        """
        Build a batch of latent vectors for the generator.

        Args:
            n_samples (int): number of vector in the batch
        """

        inputLatent = torch.randn(
            n_samples, self.config.noiseVectorDim).to(self.device)

        return inputLatent

    def getOriginalG(self):
        r"""
        Retrieve the original G network. Use this function
        when you want to modify G after the initialization
        """
        if isinstance(self.netG, nn.DataParallel):
            return self.netG.module
        return self.netG

    def getOriginalAvgG(self):
        r"""
        Retrieve the original avG network. Use this function
        when you want to modify avG after the initialization
        """
        if isinstance(self.avgG, nn.DataParallel):
            return self.avgG.module
        return self.avgG

    def getOriginalD(self):
        r"""
        Retrieve the original D network. Use this function
        when you want to modify D after the initialization
        """
        if isinstance(self.netD, nn.DataParallel):
            return self.netD.module
        return self.netD

    def getNetG(self):
        r"""
        The generator should be defined here.
        """
        pass

    def getNetD(self):
        r"""
        The discrimator should be defined here.
        """
        pass

    def getOptimizerD(self):
        r"""
        Optimizer of the discriminator.
        """
        pass

    def getOptimizerG(self):
        r"""
        Optimizer of the generator.
        """
        pass

    def getStateDict(self, saveTrainTmp=False):
        """
        Get the model's parameters
        """
        # Get the generator's state
        stateG = self.getOriginalG().state_dict()

        # Get the discrimator's state
        stateD = self.getOriginalD().state_dict()

        out_state = {'config': self.config,
                     'netG': stateG,
                     'netD': stateD}

        # Average GAN
        out_state['avgG'] = self.getOriginalAvgG().state_dict()

        if saveTrainTmp:
            out_state['tmp'] = self.trainTmp

        return out_state

    def save(self, path, saveTrainTmp=False):
        """
        Save the model at the given location.

        All parameters included in the self.config class will be saved as well.
        Args:
            - path (string): file where the model should be saved
            - saveTrainTmp (bool): set to True if you want to conserve
                                    the training parameters
        """
        torch.save(self.getStateDict(saveTrainTmp=saveTrainTmp), path)

    def load(self,
             path="",
             in_state=None,
             loadG=True,
             loadD=True,
             loadConfig=True,
             finetuning=False):
        """
        Load a model saved with the @method save() function

        Args:
            - path (string): file where the model is stored
        """

        in_state = torch.load(path)
        self.load_state_dict(in_state,
                             loadG=loadG,
                             loadD=loadD,
                             loadConfig=True,
                             finetuning=False)

    def load_state_dict(self,
                        in_state,
                        loadG=True,
                        loadD=True,
                        loadConfig=True,
                        finetuning=False):
        """
        Load a model saved with the @method save() function

        Args:
            - in_state (dict): state dict containing the model
        """

        # Step one : load the configuration
        if loadConfig:
            updateConfig(self.config, in_state['config'])
            self.lossCriterion = getattr(
                base_loss_criterions, self.config.lossCriterion)(self.device)

        # Re-initialize G and D with the loaded configuration
        buildAvG = True

        if loadG:
            self.netG = self.getNetG()
            if finetuning:
                loadPartOfStateDict(
                    self.netG, in_state['netG'], ["formatLayer"])
                self.getOriginalG().initFormatLayer(self.config.latentVectorDim)
            else:
                # Replace me by a standard loadStatedict for open-sourcing TODO
                loadStateDictCompatible(self.netG, in_state['netG'])
                if 'avgG' in in_state:
                    print("Average network found !")
                    self.buildAvG()
                    # Replace me by a standard loadStatedict for open-sourcing
                    loadStateDictCompatible(self.getOriginalAvgG(), in_state['avgG'])
                    buildAvG = False

        if loadD:
            self.netD = self.getNetD()
            if finetuning:
                loadPartOfStateDict(
                    self.netD, in_state['netD'], ["decisionLayer"])
                self.getOriginalD().initDecisionLayer(self.lossCriterion.sizeDecisionLayer)
            else:
                # Replace me by a standard loadStatedict for open-sourcing TODO
                loadStateDictCompatible(self.netD, in_state['netD'])

        elif 'tmp' in in_state.keys():
            self.trainTmp = in_state['tmp']

        # Don't forget to reset the machinery !
        self.updateSolversDevice(buildAvG)
