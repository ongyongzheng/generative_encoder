import os
import json
import pickle as pkl

import torch
import torchvision.transforms as Transforms

from ...utils.config import BaseConfig, getConfigFromDict, getDictFromConfig

class BaseTrainer():
    """
    a class to manage model training.
    """

    def __init__(self,
                 pathdb,
                 dataloader=None,
                 dbType="image",
                 targets=[""],
                 useGPU=True,
                 config=None,
                 lossIterEvaluation=1, # TODO: change back to 200
                 saveIter=5000,
                 checkPointDir=None,
                 modelLabel=""):
        """
        Initializer for all trainers
        """

        # Parameters
        # Training dataset parameters
        self.path_db = pathdb
        self.db_type = dbType
        self.targets = targets

        # set up dataloader for training
        self.dataloader = dataloader

        if config is None:
            config = {}

        # Load training configuration
        self.readTrainConfig(config)

        # Model Initialization
        self.useGPU = useGPU

        if not self.useGPU:
            self.numWorkers = 1

        # Internal state
        self.runningLoss = {}
        self.startScale = 0
        self.startIter = 0
        self.lossProfile = []

        self.initModel()

        # set checkpoint parameters
        self.checkPointDir = checkPointDir
        self.modelLabel = modelLabel
        self.saveIter = saveIter
        self.pathLossLog = None

        # Loss printing
        self.lossIterEvaluation = lossIterEvaluation

    def readTrainConfig(self, config):
        """
        load a permanent configuration describing a model.
        variables described here should remain constant through the training.
        """
        self.modelConfig = BaseConfig()
        getConfigFromDict(self.modelConfig, config, self.getDefaultConfig())

    def getDefaultConfig(self):
        """the default config to load should be implemented here"""
        pass

    def initModel(self):
        """the model should be initialized here"""
        pass

    def updateRunningLosses(self, allLosses):

        for name, value in allLosses.items():

            if name not in self.runningLoss:
                self.runningLoss[name] = [0, 0]

            self.runningLoss[name][0]+= value
            self.runningLoss[name][1]+=1

    def resetRunningLosses(self):

        self.runningLoss = {}

    def updateLossProfile(self, iter):

        nPrevIter = len(self.lossProfile[-1]["iter"])
        self.lossProfile[-1]["iter"].append(iter)

        newKeys = set(self.runningLoss.keys())
        existingKeys = set(self.lossProfile[-1].keys())

        toComplete = existingKeys - newKeys

        for item in newKeys:

            if item not in existingKeys:
                self.lossProfile[-1][item] = [None for x in range(nPrevIter)]

            value, stack = self.runningLoss[item]
            self.lossProfile[-1][item].append(value /float(stack))

        for item in toComplete:
            if item in ["scale", "iter"]:
                continue
            self.lossProfile[-1][item].append(None)

    def getDBLoader(self, scale):
        """
        Load the training dataset for the given scale.

        Args:

            - scale (int): scale at which we are working

        Returns:

            A dataset with properly resized inputs.
        """
        # prepare parameters for the dataloader
        # size
        size = self.model.getSize()

        print("size", size)
        print("loading {} dataset".format(self.db_type))

        dataset = self.dataloader.getDataset(self.path_db, self.targets, size, self.modelConfig)
        
        print("%d images detected" % int(len(dataset)))
        
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=self.modelConfig.miniBatchSize,
                                           shuffle=True, num_workers=self.model.n_devices)
    
    def inScaleUpdate(self, iter, scale, inputs_real):
        return inputs_real

    def trainOnEpoch(self,
                     dbLoader,
                     scale,
                     shiftIter=0,
                     maxIter=-1):
        pass

    def train(self):
        pass