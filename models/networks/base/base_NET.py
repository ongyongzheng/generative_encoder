# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from copy import deepcopy

import torch
import torch.nn as nn

from ...utils.config import BaseConfig , updateConfig


class BaseNET():
    """Abstract class: the basic framework for NN training.
    """

    def __init__(self,
                 config=None,
                 useGPU=True,
                 **kwargs):
        """
        Initialize Standard Base for NN training
        """

        if 'config' not in vars(self):
            self.config = BaseConfig()

        if 'trainTmp' not in vars(self):
            self.trainTmp = BaseConfig()

        self.useGPU = useGPU and torch.cuda.is_available()
        if self.useGPU:
            self.device = torch.device("cuda:0")
            self.n_devices = torch.cuda.device_count()
        else:
            self.device = torch.device("cpu")
            self.n_devices = 1

        # Output image dimension
        self.config.dimOutput = config["dimOutput"]

        # Actual learning rate
        self.config.learningRate = config["baseLearningRate"]

        # Data type
        self.config.dataType = config["dataType"]


    def test(self, input, toCPU=True):
        """
        test function to test the model
        """
        input = input.to(self.device)
        if toCPU:
            return input.cpu()
        else:
            return input

    def optimizeParameters(self):
        """
        Update the model using input_batch
        """

        # return losses
        allLosses = {}

        return allLosses

    def updateSolversDevice(self):
        """
        Move the current networks and solvers to the GPU.
        This function must be called each time the network is modified
        """
        pass

    def getStateDict(self):
        """
        Get the model's parameters
        """
        pass

    def save(self):
        """
        Save the model at the given location.
        """
        pass

    def updateConfig(self, config):
        """
        Update the object config with new inputs.

        Typically if config = {"learningRate": 0.1} only the learning rate
        will be changed.
        """
        updateConfig(self.config, config)
        self.updateSolversDevice()

    def load(self,
             path=''):
        """
        Load a model saved with the @method save() function

        Args:
            - path (string): file where the model is stored
        """
        pass

    def load_state_dict(self,
                        in_state):
        """
        Load a model saved with the @method save() function

        Args:
            - in_state (dict): state dict containing the model
        """
        pass
