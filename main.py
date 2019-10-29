import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os

from config import Config
from models.BEGAN import BEGAN

# create config
config = Config()

began = BEGAN(config)

print("model structure")
print(began.generator)
print(began.discriminator)