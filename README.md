# Generative Imaging and Image Processing via Generative Encoder

This repository implements the basic GE model found in the paper.

INSERT LINK ONCE AVAILABLE

The following models are implemented for use in the paper:

**GAN Models**
- Progressive Growing of GAN (PGAN): https://arxiv.org/pdf/1710.10196.pdf (Supports: image)

**AE Models**
- Variational Auto-Encoder (VAE): (Code Referenced From) https://github.com/sksq96/pytorch-vae, https://github.com/atinghosh/VAE-pytorch

### Dataset type legend

| Data Type | Description |
| --- | --- |
| image | Image type supports RGB and Grayscale images, in which the field "dimOutput" in the config file determines which type |

### Dataset preparation

The below code snippets assumes that your datasets are saved in the `dataset_raw` folder

- celebA cropped

Split the dataset into train and test data (either use partitions given by the webpage, or do a 9:1 split). The dataset can be downloaded from [https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

```
python datasets.py celeba_cropped dataset_raw/img_align_celeba/ -o dataset/img_align_celeba/
```

## Configuration file of a training session

The minimum configuration file for a training session is a json file with the following lines

```
{
    "pathDB": PATH_TO_YOUR_DATASET,
    "dbType": DATASET_TYPE
}
```

Where a dataset can be:
- a folder with all your images in .jpg, .png or .npy format

And a dataset type can be:
- "image"

To this you can add a "config" entry giving overrides to the standard configuration for the model. See models/trainer/standard_configurations to see all possible options. For example:

```
{
    "pathDB": PATH_TO_YOUR_DATASET,
    "dbType": DATASET_TYPE,
    "config": {"dataType": "image",
               "baseLearningRate": 0.1,
               "miniBatchSize": 22}
}
```

Will override the learning rate and the mini-batch-size. Please note that if you specify a - -baseLearningRate option in your command line, the command line will prevail. Depending on how you work you might prefer to have specific configuration files for each run or only rely on one configuration file and input your training parameters via the command line.

## Running a GE training process

![GE Model Framework](./images/GE.png)

#### Step 1: Train the GAN model

```
python train.py PGAN --restart -n celeba_pgan_clean -c config_celeba_cropped.json
```

#### Step 2: Train the VAE model

```
python train.py VAE --restart -n celeba_vae_clean -c config_celeba_cropped.json
```

#### Step 3: Run the GE model

Refer to the folder [tests/generative_encoder](./tests/generative_encoder) for the main code. Update lines 47 to 52 with output scale and iteration for model to load. Update [tests/generative_encoder/test_ge_celeba.sh](./tests/generative_encoder/test_ge_celeba.sh) with position of image to run GE on.

```
nohup bash tests/generative_encoder/test_ge_celeba.sh &> test_ge_celeba.out &
```


## CREDITS

The following sources have contributed greatly to the development of this repository:

- GAN Architecture, base code: https://github.com/facebookresearch/pytorch_GAN_zoo
- VAE References: https://github.com/sksq96/pytorch-vae, https://github.com/atinghosh/VAE-pytorch
- Generative Encoder for Image Processing: INSERT LINK ONCE AVAILABLE
- CelebA Dataset: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
