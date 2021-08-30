import torch
import torch.nn.functional as F


class BaseLossWrapper:
    """
    Loss criterion class. Must define 4 members:
    sizeDecisionLayer : size of the decision layer of the discrimator

    getCriterion : how the loss is actually computed

    !! The activation function of the discriminator is computed within the
    loss !!
    """

    def __init__(self, device):
        self.device = device

    def getCriterion(self, input, status):
        """
        Given an input tensor and its targeted status (detected as real or
        detected as fake) build the associated loss

        Args:

            - input (Tensor): decision tensor build by the model's discrimator
            - status (bool): if True -> this tensor should have been detected
                             as a real input
                             else -> it shouldn't have
        """
        pass

class AE_MSE(BaseLossWrapper):
    """
    Implements the base AE loss wrapper using MSE loss.
    """

    def __init__(self, device):
        self.generationActivation = F.tanh
        self.sizeDecisionLayer = 3

        BaseLossWrapper.__init__(self, device)

    def getCriterion(self, input, mu, var, target):

        # reconstruction loss
        recon_loss = F.mse_loss(input[:, :self.sizeDecisionLayer], target)

        return recon_loss

class VAE_MSE(BaseLossWrapper):
    """
    Implements the base VAE loss wrapper using MSE loss.
    """

    def __init__(self, device):
        self.generationActivation = None
        self.sizeDecisionLayer = 3

        BaseLossWrapper.__init__(self, device)

    def getCriterion(self, input, mu, var, target):
        kld_weight = mu.size()[1] * mu.size()[0]

        # reconstruction loss
        recon_loss = F.mse_loss(input, target)

        # kl divergence loss
        kld_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1), dim = 0)
        return recon_loss + (1 / kld_weight) * kld_loss