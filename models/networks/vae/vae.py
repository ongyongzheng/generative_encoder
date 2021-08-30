import torch.optim as optim

from ..ae.base_AE import BaseAE
from ...utils.config import BaseConfig
from ...utils.utils import finiteCheck
from .networks.vae_net import ENet, DNet


class VAE(BaseAE):
    """
    Implementation of VAE
    """
    
    def __init__(self,
                 config=None,
                 **kwargs):
        """
        Initialize VAE
        """
        if not 'config' in vars(self):
            self.config = BaseConfig()

        self.config.dimE = config["dimE"]
        self.config.dimD = config["dimD"]
        self.config.imageSize = config["imageSize"]

        BaseAE.__init__(self, config=config, **kwargs)

    def getNetD(self):

        dnet = DNet(self.config,
                    generationActivation=self.lossCriterion.generationActivation)
        return dnet

    def getNetE(self):

        enet = ENet(self.config)
        return enet

    def getOptimizerD(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.netD.parameters()),
                          betas=[0.5, 0.999], lr=self.config.learningRate)

    def getOptimizerE(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.netE.parameters()),
                          betas=[0.5, 0.999], lr=self.config.learningRate)

    def getOptimizerAE(self):
        params = []
        for param in self.netE.parameters():
            if param.requires_grad:
                params.append(param)
        for param in self.netD.parameters():
            if param.requires_grad:
                params.append(param)
        return optim.Adam(params,
                          betas=[0.5, 0.999], lr=self.config.learningRate)

    def optimizeParameters(self, input_batch, input_target):
        """
        Update the model using the given inputs for VAE.

        Args:
            input (torch.tensor): input batch of real data
            inputLabels (torch.tensor): labels of the real data

        """

        allLosses = {}

        # Retrieve the input data
        self.real_input = input_batch.to(self.device)
        self.target_input = input_target.to(self.device)

        n_samples = self.real_input.size()[0]

        # Update the encoder and discriminator
        self.optimizerAE.zero_grad()

        # #1 Predicted data (require latent and actual)
        predLatent, predMu, predVar = self.netE(self.real_input) # in VAE, latent layer consist of mu and var
        predReal = self.netD(predLatent)

        # #2 Compute loss between real and target
        lossD = self.lossCriterion.getCriterion(predReal, predMu, predVar, self.target_input)
        allLosses["lossAE"] = lossD.item()

        lossD.backward(retain_graph=True)
        finiteCheck(self.getOriginalD().parameters())
        self.optimizerAE.step()

        # Logs
        lossD = 0
        for key, val in allLosses.items():

            if key.find("lossAE") == 0:
                lossD += val

        allLosses["lossAE"] = lossD

        return allLosses

    def getSize(self):
        size = self.config.imageSize
        return (size, size)

    def test(self, input, getAvG=False, toCPU=True):
        """
        Generate some data given the input latent vector.

        Args:
            input (torch.tensor): input latent vector
        """
        input = input.to(self.device)
        if getAvG:
            if toCPU:
                return self.avgD(self.avgE(input)[0]).cpu()
            else:
                return self.avgD(self.avgE(input)[0])
        elif toCPU:
            return self.netD(self.netE(input)[0]).detach().cpu()
        else:
            return self.netD(self.netE(input)[0]).detach()

