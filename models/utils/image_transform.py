import torch
import torchvision.transforms as Transforms

import numpy as np
from PIL import Image

# The equivalent of some torchvision.transforms operations but for numpy array
# instead of PIL images
class NumpyResize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        """
        Args:

            img (np array): image to be resized

        Returns:

            np array: resized image
        """
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return np.array(img.resize(self.size, resample=Image.BILINEAR))

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class NumpyReshape(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        """
        Args:

            img (np array): image to be reshaped

        Returns:

            np array: resized image
        """
        img = Image.fromarray(np.reshape(img, self.size))

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class NumpyToTensor(object):

    def __init__(self):
        return

    def __call__(self, img):
        """
        Turn a numpy object into a tensor.
        """
        if len(img.shape) == 1:
            return torch.from_numpy(np.array(img, np.float32, copy=False))

        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)

        return Transforms.functional.to_tensor(img)
