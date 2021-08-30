"""
@author: Yong Zheng Ong
the main function for training models

1. PGAN training
nohup python train.py PGAN --restart -n celeba_pgan_clean -c config_celeba_cropped.json > celeba_cropped_pgan_clean.out &
nohup python train.py PGAN --restart -n digital_rock_train_pgan_clean -c config_digital_rock.json > digital_rock_train_pgan_clean.out &
nohup python train.py PGAN --restart -n digital_rock_train_pgan_64_clean -c config_digital_rock_64.json > digital_rock_train_pgan_64_clean.out &
nohup python train.py PGAN --restart -n digital_rock_train_pgan_128_clean -c config_digital_rock_128.json > digital_rock_train_pgan_128_clean.out &
nohup python train.py PGAN --restart -n digital_rock_train_pgan_256_clean -c config_digital_rock_256.json > digital_rock_train_pgan_256_clean.out &

2. VAE training
nohup python train.py VAE --restart -n celeba_vae_clean -c config_celeba_cropped.json > celeba_cropped_vae_clean.out &
nohup python train.py VAE --restart -n digital_rock_train_vae_clean -c config_digital_rock.json > digital_rock_train_vae_clean.out &
nohup python train.py VAE --restart -n digital_rock_train_vae_512_clean -c config_digital_rock_512.json > digital_rock_train_vae_512_clean.out &
"""
import os
import sys
import importlib
import argparse

# choosing CUDA environments
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import json # for loading of configurations

from config import * # import global variables

from models.utils.utils import loadmodule, getLastCheckPoint
from models.utils.config import getConfigOverrideFromParser, updateParserWithConfig

def getTrainer(name):
    """function used to get trainer for training"""

    if name not in AVAILABLE_MODELS:
        raise AttributeError("Invalid module name")

    return loadmodule("models.networks." + AVAILABLE_MODELS[name][0],
                      AVAILABLE_MODELS[name][1],
                      prefix='')

if __name__ == "__main__":

    # build parser
    parser = argparse.ArgumentParser(description='Training script')

    parser.add_argument('model_name', type=str,
                        help='Name of the model to launch, available models are\
                        {}. To get all possible option for a model\
                         please run train.py $MODEL_NAME -overrides'.format(", ".join(AVAILABLE_MODELS.keys())))
    parser.add_argument('--no_vis', help=' Disable all visualizations',
                        action='store_true')
    parser.add_argument('--restart', help=' If a checkpoint is detected, do \
                                           not try to load it',
                        action='store_true')
    parser.add_argument('-n', '--name', help="Model's name",
                        type=str, dest="name", default="default")
    parser.add_argument('-d', '--dir', help='Output directory',
                        type=str, dest="dir", default='output_networks')
    parser.add_argument('-c', '--config', help="Model's name",
                        type=str, dest="configPath")
    parser.add_argument('-s', '--save_iter', help="If it applies, frequence at\
                        which a checkpoint should be saved. In the case of a\
                        evaluation test, iteration to work on.",
                        type=int, dest="saveIter", default=10000)
    parser.add_argument('-e', '--eval_iter', help="If it applies, frequence at\
                        which a checkpoint should be saved",
                        type=int, dest="evalIter", default=100)

    # retrieve the model we want
    baseArgs, unknown = parser.parse_known_args()
    trainerModule = getTrainer(baseArgs.model_name)

    # build the output directory if needed
    if not os.path.isdir(baseArgs.dir):
        os.mkdir(baseArgs.dir)

    # add overrides to the parser
    parser = updateParserWithConfig(parser, trainerModule._defaultConfig)
    kwargs = vars(parser.parse_args())
    configOverride = getConfigOverrideFromParser(
        kwargs, trainerModule._defaultConfig)

    if kwargs['overrides']:
        parser.print_help()
        sys.exit()

    # load checkpoint data
    modelLabel = kwargs["name"]
    restart = kwargs["restart"]
    checkPointDir = os.path.join(kwargs["dir"], modelLabel)
    checkPointData = getLastCheckPoint(checkPointDir, modelLabel)

    if not os.path.isdir(checkPointDir):
        os.mkdir(checkPointDir)

    # training configurations
    configPath = kwargs.get("configPath", None)
    if configPath is None:
        raise ValueError("You need to input a configuration file")

    with open(kwargs["configPath"], 'rb') as file:
        trainingConfig = json.load(file)

    # model configuration
    modelConfig = trainingConfig.get("config", {})
    for item, val in configOverride.items():
        modelConfig[item] = val
    trainingConfig["config"] = modelConfig

    # setup visualization module
    vis_module = None
    if baseArgs.no_vis:
        print("Visualization disabled")
    else:
        if trainingConfig["dbType"] in AVAILABLE_DBTYPES:
            vis_module = importlib.import_module("visualization.{}_visualizer".format(trainingConfig["dbType"]))
        else:
            raise NotImplementedError("Visualizer for given dbType {} is not implemented".format(trainingConfig["dbType"]))

    # setup dataloader module
    dat_module = None
    if trainingConfig["dbType"] in AVAILABLE_DBTYPES:
        dat_module = importlib.import_module("dataloader.{}_dataloader".format(trainingConfig["dbType"]))
    else:
        raise NotImplementedError("Dataloader for given dbType {} is not implemented".format(trainingConfig["dbType"]))

    print("Running " + baseArgs.model_name)

    # path to the image dataset
    pathDB = trainingConfig["pathDB"]
    trainingConfig.pop("pathDB", None)

    # load trainer
    Trainer = trainerModule(pathDB,
                            useGPU=True,
                            visualisation=vis_module,
                            dataloader=dat_module,
                            lossIterEvaluation=kwargs["evalIter"],
                            checkPointDir=checkPointDir,
                            saveIter=kwargs["saveIter"],
                            modelLabel=modelLabel,
                            **trainingConfig)

    # if a checkpoint is found, load it
    if not restart and checkPointData is not None:
        trainConfig, pathModel, pathTmpData = checkPointData
        print(f"Model found at path {pathModel}, pursuing the training")
        Trainer.loadSavedTraining(pathModel, trainConfig, pathTmpData)

    Trainer.train()
