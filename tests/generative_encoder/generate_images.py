"""
@author: Yong Zheng Ong
This code implements the GE framework
"""
import os
import sys
import importlib
import argparse
import json
import pickle as pkl

import numpy as np
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
import torchvision.transforms as Transforms

from models.utils.utils import getLastCheckPoint, loadmodule, \
    parse_state_name, getNameAndPackage
from models.utils.image_transform import NumpyResize, NumpyToTensor
from models.utils.config import getConfigOverrideFromParser, \
    updateParserWithConfig

from dataloader.image_dataloader import pil_loader, standardTransform

import json

def test(parser):
    # load some constants
    args = parser.parse_args()
    with_ae = args.with_ae
    evaluation_id = args.evaluation_id
    if not os.path.exists('./results/{}'.format(evaluation_id)):
        os.makedirs('./results/{}'.format(evaluation_id))

    if args.evaluation_name == 'celeba':
        key = [f for f in sorted(os.listdir('dataset/img_align_celeba_test_cropped')) if f[-3:] == 'jpg'][args.position]
        image_to_test = 'dataset/img_align_celeba_test_cropped/{}.jpg'.format(key)

        # load config data
        with open('config_celeba_cropped.json', 'rb') as file:
            trainingConfig = json.load(file)

        scale_gan = 5
        iter_gan = 200000

        if with_ae:
            scale_ae = 0
            iter_ae = 1536000

    if not os.path.exists('./results/plots'):
        os.makedirs('./results/plots')

    # get required fields
    modelConfig = trainingConfig.get("config")
    dimOutput = modelConfig["dimOutput"]

    # load GAN model
    # gan parameters -----
    print('loading GAN model...')
    default_dir = "output_networks"
    config_gan = {
        'name' : args.gan_name, # model name (str)
        'module' : 'PGAN',
        'scale' : scale_gan, # scale to evaluate (int)
        'iter' : iter_gan, # iteration to evaluate (int)
    }
    # get checkpoint data
    checkPointDir = os.path.join(default_dir, config_gan["name"])
    checkpointData = getLastCheckPoint(
        checkPointDir, config_gan["name"], scale=config_gan["scale"], iter=config_gan["iter"])

    if checkpointData is None:
        print(config_gan["scale"], config_gan["iter"])
        if config_gan["scale"] is not None or config_gan["iter"] is not None:
            raise FileNotFoundError("Not checkpoint found for model "
                                    + config_gan["name"] + " at directory " + default_dir +
                                    " for scale " + str(config_gan["scale"]) +
                                    " at iteration " + str(config_gan["iter"]))
        raise FileNotFoundError(
            "Not checkpoint found for model " + config_gan["name"] + " at directory "
            + default_dir)

    modelConfig, pathModel, _ = checkpointData
    with open(modelConfig, 'rb') as file:
            configData = json.load(file)

    modelPackage, modelName = getNameAndPackage(config_gan["module"])
    modelType = loadmodule(modelPackage, modelName)

    gan_model = modelType(useGPU=True,
                          storeAVG=True,
                          config=configData)

    if config_gan["scale"] is None or config_gan["iter"] is None:
        _, config_gan["scale"], config_gan["iter"] = parse_state_name(pathModel)

    print("Checkpoint found at scale %d, iter %d" % (config_gan["scale"], config_gan["iter"]))
    gan_model.load(pathModel, loadD=False)

    if with_ae:
        # load AE model
        # ae parameters -----
        print('loading AE model...')
        default_dir = "output_networks"
        config_ae = {
            'name' : args.ae_name, # model name (str)
            'module' : 'VAE',
            'scale' : scale_ae, # scale to evaluate (int)
            'iter' : iter_ae, # iteration to evaluate (int)
        }
        # get checkpoint data
        checkPointDir = os.path.join(default_dir, config_ae["name"])
        checkpointData = getLastCheckPoint(
            checkPointDir, config_ae["name"], scale=config_ae["scale"], iter=config_ae["iter"])

        if checkpointData is None:
            print(config_ae["scale"], config_ae["iter"])
            if config_ae["scale"] is not None or config_ae["iter"] is not None:
                raise FileNotFoundError("Not checkpoint found for model "
                                        + config_ae["name"] + " at directory " + default_dir +
                                        " for scale " + str(config_ae["scale"]) +
                                        " at iteration " + str(config_ae["iter"]))
            raise FileNotFoundError(
                "Not checkpoint found for model " + config_gan["name"] + " at directory "
                + default_dir)

        modelConfig, pathModel, _ = checkpointData
        with open(modelConfig, 'rb') as file:
                configData = json.load(file)

        modelPackage, modelName = getNameAndPackage(config_ae["module"])
        modelType = loadmodule(modelPackage, modelName)

        ae_model = modelType(useGPU=True,
                             storeAVG=False,
                             config=configData)

        if config_ae["scale"] is None or config_ae["iter"] is None:
            _, config_ae["scale"], config_ae["iter"] = parse_state_name(pathModel)

        print("Checkpoint found at scale %d, iter %d" % (config_ae["scale"], config_ae["iter"]))
        ae_model.load(pathModel, loadD=False)

    # load the image
    size = gan_model.getSize()
    print("size", size)

    # build standard transform for images
    inTransform, outTransform = standardTransform(size, dimOutput)

    image_real = inTransform(pil_loader(image_to_test)).view(1,dimOutput,*size).to('cuda:0')
    print("image shape:", image_real.size(), "| min:", np.min(image_real.detach().cpu().numpy()), "| max:", np.max(image_real.detach().cpu().numpy()))
    if with_ae:
        encoded_real = ae_model.netE(image_real)[0].detach()
        print("vector shape:", encoded_real.size(), "| min:", np.min(encoded_real.cpu().numpy()), "| max:", np.max(encoded_real.cpu().numpy()))

    transform = outTransform

    image_real_save = torch.clamp(image_real.cpu(), min=-1, max=1)

    transform(image_real_save[0]).save("./results/{}/{}_real.jpg".format(evaluation_id, key))

    num_iter = 10000
    num_to_find = 1000
    # get random start vector from 1000 points
    start_choices = gan_model.buildNoiseData(num_to_find)

    start = start_choices[0:1,:]
    score = 999999999999999999999999999
    # find best start vector
    for i in range(num_to_find):
        start_generation = gan_model.avgG(start_choices[i:i+1,:])
        error = torch.nn.MSELoss()(start_generation, image_real).cpu().detach().item()
        if error < score:
            score = error
            start = start_choices[i:i+1,:]

    print("final best: ", score, " | start shape: ", start.shape)

    class start_vector(torch.nn.Module):
        def __init__(self, start):
            super(start_vector, self).__init__()
            self.vector = torch.nn.Parameter(start, requires_grad=True)

        def forward(self):
            return self.vector

    sv = start_vector(start).to('cuda:0')
    ge_optimizer = torch.optim.Adam(sv.parameters(), lr=0.01)

    best = 100
    best_vector = sv.vector.cpu().detach().numpy()
    errors = []

    threshold = 0.000
    i = 0
    while best > threshold:
        ge_optimizer.zero_grad()
        output = gan_model.avgG(sv())

        if with_ae:
            encoded = ae_model.netE(output)[0]
            reconstruction = torch.nn.MSELoss()(output, image_real)
            encoded = torch.nn.MSELoss()(encoded, encoded_real)
            loss = encoded
        else:
            reconstruction = torch.nn.MSELoss()(output, image_real)
            loss = reconstruction

        loss.backward()
        ge_optimizer.step()

        errors.append(reconstruction.cpu().detach().item())

        if (i+1) % 100 == 0:
            # to save
            if best is None:
                best = reconstruction.cpu().detach().item()
                transform(torch.clamp(output.detach().cpu(), min=-1, max=1)[0]).save("./results/{}/{}_ge.jpg".format(evaluation_id, key))
                best_vector = sv.vector.cpu().detach().numpy()
            elif best >= reconstruction.cpu().detach().item():
                best = reconstruction.cpu().detach().item()
                transform(torch.clamp(output.detach().cpu(), min=-1, max=1)[0]).save("./results/{}/{}_ge.jpg".format(evaluation_id, key))
                best_vector = sv.vector.cpu().detach().numpy()
        i += 1
        if i == num_iter:
            break
    print("best: ", best)

    np.savetxt("./results/plots/{}_{}.csv".format(evaluation_id, key), errors, delimiter =", ", fmt ='% s')
    np.save("./results/{}/{}_vector.npy".format(evaluation_id, key), best_vector)
