import os
import json
import pickle as pkl

import torch

from ..base.base_trainer import BaseTrainer
from ...utils.config import BaseConfig, getConfigFromDict, getDictFromConfig

class AETrainer(BaseTrainer):
    """
    A class managing an AE training. Logs, chekpoints,
    visualization, and number iterations are managed here.
    """

    def __init__(self,
                 pathdb,
                 visualisation=None,
                 **kwargs):
        """
        Initializer for all AE trainers
        """
        BaseTrainer.__init__(self, pathdb, **kwargs)

        # set up visualisation for AE training
        self.visualisation = visualisation
        self.tokenWindowReconstructedSmooth = None
        self.tokenWindowReconstructed = None
        self.tokenWindowOriginal = None
        self.tokenWindowLosses = None

        self.nDataVisualization = 16
        
        # additional checkpoint paths
        if self.checkPointDir is not None:
            self.pathLossLog = os.path.abspath(os.path.join(self.checkPointDir,
                                                            self.modelLabel
                                                            + '_losses.pkl'))
            self.pathRefVector = os.path.abspath(os.path.join(self.checkPointDir,
                                                              self.modelLabel
                                                              + '_refVectors.pt'))

    def initModel(self):
        """
        Initialize the AE model
        """
        pass

    def loadSavedTraining(self,
                          pathModel,
                          pathTrainConfig,
                          pathTmpConfig,
                          loadE=True,
                          loadD=True,
                          finetune=False):
        """
        Load a given checkpoint.

        Args:

            - pathModel (string): path to the file containing the model
                                 structure (.pt)
            - pathTrainConfig (string): path to the reference configuration
                                        file of the training. WARNING: this
                                        file must be compatible with the one
                                        pointed by pathModel
            - pathTmpConfig (string): path to the temporary file describing the
                                      state of the training when the checkpoint
                                      was saved. WARNING: this file must be
                                      compatible with the one pointed by
                                      pathModel
        """

        # Load the temp configuration
        tmpPathLossLog = None
        tmpConfig = {}

        if pathTmpConfig is not None:
            tmpConfig = json.load(open(pathTmpConfig, 'rb'))
            self.startScale = tmpConfig["scale"]
            self.startIter = tmpConfig["iter"]
            self.runningLoss = tmpConfig.get("runningLoss", {})

            tmpPathLossLog = tmpConfig.get("lossLog", None)

        if tmpPathLossLog is None:
            self.lossProfile = [
                {"iter": [], "scale": self.startScale}]
        elif not os.path.isfile(tmpPathLossLog):
            print("WARNING : couldn't find the loss logs at " +
                  tmpPathLossLog + " resetting the losses")
            self.lossProfile = [
                {"iter": [], "scale": self.startScale}]
        else:
            self.lossProfile = pkl.load(open(tmpPathLossLog, 'rb'))
            self.lossProfile = self.lossProfile[:(self.startScale + 1)]

            if self.lossProfile[-1]["iter"][-1] > self.startIter:
                indexStop = next(x[0] for x in enumerate(self.lossProfile[-1]["iter"])
                                 if x[1] > self.startIter)
                self.lossProfile[-1]["iter"] = self.lossProfile[-1]["iter"][:indexStop]

                for item in self.lossProfile[-1]:
                    if isinstance(self.lossProfile[-1][item], list):
                        self.lossProfile[-1][item] = \
                            self.lossProfile[-1][item][:indexStop]

        # Read the training configuration
        if not finetune:
            trainConfig = json.load(open(pathTrainConfig, 'rb'))
            self.readTrainConfig(trainConfig)

        # Re-initialize the model
        self.initModel()
        self.model.load(pathModel,
                        loadE=loadE,
                        loadD=loadD,
                        finetuning=finetune)

    def getDefaultConfig(self):
        pass

    def resetVisualization(self, nDataVisualization):

        self.nDataVisualization = nDataVisualization

    def saveBaseConfig(self, outPath):
        """
        Save the model basic configuration (the part that doesn't change with
        the training's progression) at the given path
        """

        outConfig = getDictFromConfig(
            self.modelConfig, self.getDefaultConfig())

        with open(outPath, 'w') as fp:
            json.dump(outConfig, fp, indent=4)

    def saveCheckpoint(self, outDir, outLabel, scale, iter):
        """
        Save a checkpoint at the given directory. Please not that the basic
        configuration won't be saved.

        This function produces 2 files:
        outDir/outLabel_tmp_config.json -> temporary config
        outDir/outLabel -> networks' weights

        And update the two followings:
        outDir/outLabel_losses.pkl -> losses util the last registered iteration
        outDir/outLabel_refVectors.pt -> reference vectors for visualization
        """
        pathModel = os.path.join(outDir, outLabel + ".pt")
        self.model.save(pathModel)

        # Tmp Configuration
        pathTmpConfig = os.path.join(outDir, outLabel + "_tmp_config.json")
        outConfig = {'scale': scale,
                     'iter': iter,
                     'lossLog': self.pathLossLog,
                     'refVectors': self.pathRefVector,
                     'runningLoss': self.runningLoss}

        # Save the reference vectors
        torch.save(self.refVectorTarget, open(self.pathRefVector, 'wb'))

        with open(pathTmpConfig, 'w') as fp:
            json.dump(outConfig, fp, indent=4)

        if self.pathLossLog is None:
            raise AttributeError("Logging mode disabled")

        if self.pathLossLog is not None:
            pkl.dump(self.lossProfile, open(self.pathLossLog, 'wb'))

        if self.visualisation is not None:
            ref_g = self.model.test(self.refVectorVisualization)
            imgSize = max(128, ref_g.size()[2])
            self.visualisation.saveTensor(ref_g, (imgSize, imgSize),
                                          os.path.join(outDir, outLabel + '.jpg'))

    def sendToVisualization(self, refVectorReal, refVectorTarget, scale, label=None):
        """
        Send the images generated from some reference latent vectors and a
        bunch of real examples from the dataset to the visualisation tool.
        """
        imgSize = max(128, refVectorReal.size()[2])
        envLabel = self.modelLabel + "_training"

        label = self.modelLabel

        ref_g_smooth = self.model.test(refVectorReal, True) # test with running average generator
        self.tokenWindowReconstructedSmooth = \
            self.visualisation.publishTensors(ref_g_smooth,
                                              (imgSize, imgSize),
                                              label + " smooth",
                                              self.tokenWindowReconstructedSmooth,
                                              env=envLabel)

        ref_g = self.model.test(refVectorReal, False) # test without running average generator
        
        self.tokenWindowReconstructed = \
            self.visualisation.publishTensors(ref_g,
                                              (imgSize, imgSize),
                                              label + " reconstructed",
                                              self.tokenWindowReconstructed,
                                              env=envLabel)
        self.tokenWindowOriginal = \
            self.visualisation.publishTensors(refVectorTarget,
                                              (imgSize, imgSize),
                                              label + " original",
                                              self.tokenWindowOriginal,
                                              env=envLabel)
        self.tokenWindowLosses = \
            self.visualisation.publishLoss(self.lossProfile[scale],
                                           self.modelLabel,
                                           self.tokenWindowLosses,
                                           env=envLabel)

    def trainOnEpoch(self,
                     dbLoader,
                     scale,
                     shiftIter=0,
                     maxIter=-1):
        """
        Train the model on one epoch.

        Args:

            - dbLoader (DataLoader): dataset on which the training will be made
            - scale (int): scale at which is the training is performed
            - shiftIter (int): shift to apply to the iteration index when
                               looking for the next update of the alpha
                               coefficient
            - maxIter (int): if > 0, iteration at which the training should stop

        Returns:

            True if the training went smoothly
            False if a diverging behavior was detected and the training had to
            be stopped
        """

        i = shiftIter

        for _, data in enumerate(dbLoader, 0):

            if len(self.targets) == 1:
                inputs_real = data
                inputs_target = data
            elif len(self.targets) == 2:
                inputs_real, inputs_target = data

            # set visualization vector
            self.refVectorVisualization = inputs_real
            self.refVectorTarget = inputs_target

            if inputs_real.size()[0] < self.modelConfig.miniBatchSize:
                continue

            # Additionnal updates inside a scale
            inputs_real = self.inScaleUpdate(i, scale, inputs_real)

            allLosses = self.model.optimizeParameters(inputs_real, inputs_target)

            self.updateRunningLosses(allLosses)

            i += 1

            # Regular evaluation
            if i % self.lossIterEvaluation == 0:

                # Reinitialize the losses
                self.updateLossProfile(i)

                print('[%d : %6d] loss E : %.3f' % (scale, i,
                      self.lossProfile[-1]["lossAE"][-1]))

                self.resetRunningLosses()

                if self.visualisation is not None:
                    self.sendToVisualization(inputs_real, inputs_target, scale)

            if self.checkPointDir is not None:
                if i % self.saveIter == 0:
                    labelSave = self.modelLabel + ("_s%d_i%d" % (scale, i))
                    self.saveCheckpoint(self.checkPointDir,
                                        labelSave, scale, i)

            if i == maxIter:
                return True

        return True

