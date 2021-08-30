import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as Transforms

from models.utils.image_transform import NumpyResize, NumpyToTensor, NumpyReshape

class ImageDataset(Dataset):
    """
    A dataset class adapted to image folders.
    It loads the images.
    """

    def __init__(self,
                 pathdb,
                 target=[""],
                 transform=None):
        """
        initializer for the image dataset
        """

        # set parameters
        self.pathdb = pathdb
        self.target = target
        self.transform = transform

        self.listImg = [[imgName for imgName in os.listdir(os.path.join(pathdb, target_name))
                        if os.path.splitext(imgName)[1] in [".jpg", ".png",
                                                            ".npy"]] for target_name in self.target]

        if len(self.listImg[0]) == 0:
            raise AttributeError("Empty dataset found")

        print("%d images found" % len(self.listImg[0]))

    def __len__(self):
        return len(self.listImg[0])

    def __getitem__(self, idx):
        images = []

        for i in range(len(self.target)):
            imgName = self.listImg[i][idx]
            imgPath = os.path.join(self.pathdb, self.target[i], imgName)
            img_i = pil_loader(imgPath)

            if self.transform is not None:
                img_i = self.transform(img_i)

            images.append(img_i)

        if len(images) == 1:
            return images[0]
        elif len(images) == 2:
            return images[0], images[1]

def pil_loader(path, color=True):
    """
    loads the image from the given path
    """
    imgExt = os.path.splitext(path)[1]
    if imgExt == ".npy":
        img = np.load(path)[0]
        return np.swapaxes(np.swapaxes(img, 0, 2), 0, 1)

    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if color:
            return img.convert('RGB')
        else:
            return img.convert('L')

def getDataset(path_db, targets, size, modelConfig):
    """
    get image dataset for scale and size

    Inputs:
        size [] - size array of the dataset
    """
    # in image datasets, we do resize, numpy to tensor, then normalize
    transformList = [NumpyResize(size),
                     NumpyToTensor()]

    # handle rgb vs grayscale images
    if modelConfig.dimOutput == 1:
        transformList = [Transforms.Grayscale(1)] + transformList + [Transforms.Normalize((0.5), (0.5))]
    else:
        transformList = transformList + [Transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    transform = Transforms.Compose(transformList)

    return ImageDataset(path_db,
                        target=targets,
                        transform=transform)