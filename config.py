"""
file containing constants for configurations
"""

# list of available models implemented currently
AVAILABLE_MODELS = {
    # GAN Based Models
    "PGAN": ("progressive_gan.trainer", "ProgressiveGANTrainer"),
    # AE Based Models
    "VAE": ("vae.trainer", "VAETrainer")
}

# list of available fields for dbType implemented currently
AVAILABLE_DBTYPES = (
    "image"
)