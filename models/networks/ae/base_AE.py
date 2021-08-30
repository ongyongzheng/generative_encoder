from copy import deepcopy

import torch
import torch.nn as nn

from ..base.base_NET import BaseNET
from ...utils.config import BaseConfig, updateConfig
from .loss_criterions import base_loss_criterions
from ...utils.utils import loadPartOfStateDict, finiteCheck, \
    loadStateDictCompatible

class BaseAE(BaseNET):
    """Abstract class: the basic framework for AE training.
    """

    def __init__(self,
                 config=None,
                 **kwargs):
        """
        Initialize the BaseAE

        In addition to base parameters, AE additionally requires
        1. lossMode
        """

        BaseNET.__init__(self, config=config, **kwargs)

        self.valid_losses = ['AE_MSE', 'VAE_MSE']

        if config["lossMode"] not in self.valid_losses:
            raise ValueError(
                "lossMode should be one of the following : {}".format(self.valid_losses))

        # Latent vector dimension
        self.config.latentVectorDim = config["dimLatentVector"]

        # Loss criterion
        self.config.lossCriterion = config["lossMode"]
        self.lossCriterion = getattr(
            base_loss_criterions, config["lossMode"])(self.device)

        # Initialize the encoder and the decoder
        self.netD = self.getNetD()
        self.netE = self.getNetE()

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
                return self.avgD(self.avgE(input)).cpu()
            else:
                return self.avgD(self.avgE(input))
        elif toCPU:
            return self.netD(self.netE(input)).detach().cpu()
        else:
            return self.netD(self.netE(input)).detach()

    def buildAvG(self):
        """
        Create and upload a moving average encoder and decoder.
        """
        # generate decoder
        self.avgD = deepcopy(self.getOriginalD())
        for param in self.avgD.parameters():
            param.requires_grad = False

        if self.useGPU:
            self.avgD = nn.DataParallel(self.avgD)
            self.avgD.to(self.device)

        # generate encoder
        self.avgE = deepcopy(self.getOriginalE())
        for param in self.avgE.parameters():
            param.requires_grad = False

        if self.useGPU:
            self.avgE = nn.DataParallel(self.avgE)
            self.avgE.to(self.device)

    def optimizeParameters(self, input_batch, input_target):
        """
        Update the model using the given inputs. The optimization flow should be described here.

        Args:
            input (torch.tensor): input batch of real data
            inputLabels (torch.tensor): labels of the real data

        """
        pass

    def updateSolversDevice(self, buildAvG=True):
        """
        Move the current networks and solvers to the GPU.
        This function must be called each time netE or netD is modified
        """
        if buildAvG:
            self.buildAvG()

        if not isinstance(self.netD, nn.DataParallel) and self.useGPU:
            self.netD = nn.DataParallel(self.netD)
        if not isinstance(self.netE, nn.DataParallel) and self.useGPU:
            self.netE = nn.DataParallel(self.netE)

        self.netD.to(self.device)
        self.netE.to(self.device)

        self.optimizerD = self.getOptimizerD()
        self.optimizerE = self.getOptimizerE()
        self.optimizerAE = self.getOptimizerAE()

        self.optimizerD.zero_grad()
        self.optimizerE.zero_grad()
        self.optimizerAE.zero_grad()

    def getOriginalE(self):
        r"""
        Retrieve the original E network. Use this function
        when you want to modify E after the initialization
        """
        if isinstance(self.netE, nn.DataParallel):
            return self.netE.module
        return self.netE

    def getOriginalD(self):
        r"""
        Retrieve the original D network. Use this function
        when you want to modify D after the initialization
        """
        if isinstance(self.netD, nn.DataParallel):
            return self.netD.module
        return self.netD

    def getOriginalAvgD(self):
        r"""
        Retrieve the original avG network. Use this function
        when you want to modify avG after the initialization
        """
        if isinstance(self.avgD, nn.DataParallel):
            return self.avgD.module
        return self.avgD

    def getOriginalAvgE(self):
        r"""
        Retrieve the original avG network. Use this function
        when you want to modify avG after the initialization
        """
        if isinstance(self.avgE, nn.DataParallel):
            return self.avgE.module
        return self.avgE

    def getNetE(self):
        r"""
        The encoder should be defined here.
        """
        pass

    def getNetD(self):
        r"""
        The discrimator should be defined here.
        """
        pass

    def getOptimizerD(self):
        r"""
        Optimizer of the decoder.
        """
        pass

    def getOptimizerE(self):
        r"""
        Optimizer of the encoder.
        """
        pass

    def getOptimizerAE(self):
        r"""
        Optimizer of the autoencoder.
        """
        pass

    def getStateDict(self, saveTrainTmp=False):
        """
        Get the model's parameters
        """
        # Get the generator's state
        stateE = self.getOriginalE().state_dict()

        # Get the discrimator's state
        stateD = self.getOriginalD().state_dict()

        out_state = {'config': self.config,
                     'netE': stateE,
                     'netD': stateD}

        # Average AE
        out_state['avgE'] = self.getOriginalAvgE().state_dict()
        out_state['avgD'] = self.getOriginalAvgD().state_dict()

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
             loadE=True,
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
                             loadE=loadE,
                             loadD=loadD,
                             loadConfig=True,
                             finetuning=False)

    def load_state_dict(self,
                        in_state,
                        loadE=True,
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

        # Re-initialize E and D with the loaded configuration
        buildAvG = True

        if loadE:
            self.netE = self.getNetE()
            if finetuning:
                loadPartOfStateDict(
                    self.netE, in_state['netE'], ["formatLayer"])
                self.getOriginalE().initFormatLayer(self.config.latentVectorDim)
            else:
                # Replace me by a standard loadStatedict for open-sourcing TODO
                loadStateDictCompatible(self.netE, in_state['netE'])
                if 'avgE' in in_state:
                    print("Average network found !")
                    self.buildAvG()
                    # Replace me by a standard loadStatedict for open-sourcing
                    loadStateDictCompatible(self.getOriginalAvgE(), in_state['avgE'])
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
                if 'avgD' in in_state:
                    print("Average network found !")
                    self.buildAvG()
                    # Replace me by a standard loadStatedict for open-sourcing
                    loadStateDictCompatible(self.getOriginalAvgD(), in_state['avgD'])
                    buildAvG = False

        elif 'tmp' in in_state.keys():
            self.trainTmp = in_state['tmp']

        # Don't forget to reset the machinery !
        self.updateSolversDevice(buildAvG)
