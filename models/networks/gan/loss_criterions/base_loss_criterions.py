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

class WGANGP(BaseLossWrapper):
    """
    Paper WGANGP loss : linear activation for the generator.
    https://arxiv.org/pdf/1704.00028.pdf
    """

    def __init__(self, device):

        self.generationActivation = None
        self.sizeDecisionLayer = 1

        BaseLossWrapper.__init__(self, device)

    def getCriterion(self, input, status):
        if status:
            return -input[:, 0].sum()
        return input[:, 0].sum()