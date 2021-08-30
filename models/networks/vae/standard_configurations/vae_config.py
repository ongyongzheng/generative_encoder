from ....utils.config import BaseConfig

# Default configuration for VAETrainer
_C = BaseConfig()

############################################################
# Data Type
_C.dataType = 'image'

# Image Size
_C.imageSize = 128

# Mini batch size
_C.miniBatchSize = 16

# Dimension of the latent vector
_C.dimLatentVector = 256

# Dimension of the output image
_C.dimOutput = 3

# Dimension of the encoder
_C.dimE = 32

# Dimension of the discrimator
_C.dimD = 32

# Loss mode
_C.lossMode = 'VAE_MSE'

# Base learning rate
_C.baseLearningRate = 0.0002

# Number of epochs
_C.nEpoch = 500