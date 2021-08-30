"""
@author: Yong Zheng Ong
the main visualizer package for images
"""

import visdom
import torch
import torchvision.transforms as Transforms
import torchvision.utils as vutils
import numpy as np
import random

vis = visdom.Visdom()


def resizeTensor(data, out_size_image):
    """
    postprocess image tensor before publishing to visdom

    Inputs:
        data           [Tensor: (batch_size, num_channels, height, width)] - tensor array containing image (raw output) to be displayed.
        out_size_image [Tuple: (height, width)]                            - expected height and width of the visualization window.
    """

    out_data_size = (data.size()[0], data.size()[
                     1], out_size_image[0], out_size_image[1])

    outdata = torch.empty(out_data_size)
    data = torch.clamp(data, min=-1, max=1)

    interpolationMode = 0
    if out_size_image[0] < data.size()[0] and out_size_image[1] < data.size()[1]:
        interpolationMode = 2

    # handle single channel image
    if out_data_size[1] == 1:
        transform = Transforms.Compose([Transforms.Normalize((-1.), (2)),
                                        Transforms.ToPILImage(),
                                        Transforms.Resize(
                                            out_size_image, interpolation=interpolationMode),
                                        Transforms.ToTensor()])
    else:
        transform = Transforms.Compose([Transforms.Normalize((-1., -1., -1.), (2, 2, 2)),
                                        Transforms.ToPILImage(),
                                        Transforms.Resize(
                                            out_size_image, interpolation=interpolationMode),
                                        Transforms.ToTensor()])

    for img in range(out_data_size[0]):
        outdata[img] = transform(data[img])

    return outdata


def publishTensors(data, out_size_image, caption="", window_token=None, env="main", nrow=16):
    """
    publish the tensors to visdom images

    Inputs:
        data           [Tensor: (batch_size, num_channels, height, width)] - tensor array containing image (raw output) to be displayed.
        out_size_image [Tuple: (height, width)]                            - expected height and width of the visualization window.
    """
    global vis

    # perform post processing for the image to be published
    outdata = resizeTensor(data, out_size_image)

    return vis.images(outdata, opts=dict(caption=caption), win=window_token, env=env, nrow=nrow)


def saveTensor(data, out_size_image, path):
    """
    save the tensors to path

    Inputs:
        data           [Tensor: (batch_size, num_channels, height, width)] - tensor array containing image (raw output) to be displayed.
        out_size_image [Tuple: (height, width)]                            - expected height and width of the visualization window.
        path           [String]                                            - path to save the image to.
    """

    # perform post processing for the image to be saved
    outdata = resizeTensor(data, out_size_image)

    vutils.save_image(outdata, path)


def publishLoss(data, name="", window_tokens=None, env="main"):
    """
    publish the loss to visdom images

    Inputs:
        data [Dict] - dictionary containing key (title) and value (list of loss) to be displayed.
    """

    if window_tokens is None:
        window_tokens = {key: None for key in data}

    for key, plot in data.items():

        # skip metadata not needed to be printed
        if key in ("scale", "iter"):
            continue

        nItems = len(plot)
        inputY = np.array([plot[x] for x in range(nItems) if plot[x] is not None])
        inputX = np.array([data["iter"][x] for x in range(nItems) if plot[x] is not None])

        opts = {'title': key + (' scale %d loss over time' % data["scale"]),
                'legend': [key], 'xlabel': 'iteration', 'ylabel': 'loss'}

        window_tokens[key] = vis.line(X=inputX, Y=inputY, opts=opts,
                                      win=window_tokens[key], env=env)

    return window_tokens


def delete_env(name):

    vis.delete_env(name)
