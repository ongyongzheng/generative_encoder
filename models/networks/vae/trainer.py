import os

from .standard_configurations.vae_config import _C
from .vae import VAE
from ..ae.ae_trainer import AETrainer

class VAETrainer(AETrainer):
    """
    A class managing a VAE training. Logs, chekpoints,
    visualization, and number iterations are managed here.
    """

    _defaultConfig = _C

    def getDefaultConfig(self):
        return VAETrainer._defaultConfig

    def __init__(self,
                 pathdb,
                 **kwargs):
        """
        Initializer for VAE
        """

        AETrainer.__init__(self, pathdb, **kwargs)

        self.lossProfile.append({"iter": [], "scale": 0})

    def initModel(self):
        """
        Initialize the VAE model
        """
        config = {key: value for key, value in vars(self.modelConfig).items()}
        self.model = VAE(useGPU=self.useGPU, config=config)

    def train(self):

        shift = 0
        if self.startIter >0:
            shift+= self.startIter

        if self.checkPointDir is not None:
            pathBaseConfig = os.path.join(self.checkPointDir, self.modelLabel
                                          + "_train_config.json")
            self.saveBaseConfig(pathBaseConfig)

        maxShift = int(self.modelConfig.nEpoch * len(self.getDBLoader(0)))

        for epoch in range(self.modelConfig.nEpoch):
            dbLoader = self.getDBLoader(0)
            self.trainOnEpoch(dbLoader, 0, shiftIter=shift)

            shift += len(dbLoader)

            if shift > maxShift:
                break

        label = self.modelLabel + ("_s%d_i%d" %
                                   (0, shift))
        self.saveCheckpoint(self.checkPointDir,
                            label, 0, shift)