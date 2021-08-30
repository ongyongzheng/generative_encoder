# Model Zoo for Ong Yong Zheng

This repository implements a model zoo.

The following models are currently supported:

**GAN Models**
- Progressive Growing of GAN (PGAN): https://arxiv.org/pdf/1710.10196.pdf (Supports: image, 1d_signal)
- Wasserstein GAN (WGAN): (Supports: image, 1d_signal, vector)
- Differentiable Vector Graphics (VGAN): https://github.com/BachiLi/diffvg, https://people.csail.mit.edu/tzumao/diffvg/ (supported by WGAN) (Supports: vector)

**AE Models**
- Variational Auto-Encoder (VAE): (Code Referenced From) https://github.com/sksq96/pytorch-vae, https://github.com/atinghosh/VAE-pytorch

**GE Framework**
- Digital Rock Image Inpainting using GANs: https://www.researchgate.net/publication/342464064_Digital_Rock_Image_Inpainting_using_GANs

### Model Banks

1. Differentiable Vector Graphics

- Standard Model: [VGAN](models/networks/vector_gan)

~~~
Inputs: noise vector
Outputs: vector coordinate information
~~~

### Dataset type legend

| Data Type | Description |
| --- | --- |
| image | Image type supports RGB and Grayscale images, in which the field "dimOutput" in the config file determines which type |
| 1d_signal | 1D signals are 1 dimensional signals that will be directly used for training |
| vector | Vector images will support vector generation |

### Dataset preparation

The below code snippets assumes that your datasets are saved in the `dataset_raw` folder

- celebA cropped

```
python datasets.py celeba_cropped dataset_raw/img_align_celeba/ -o dataset/img_align_celeba/
```

- musdb

```
python datasets.py musdb dataset_raw/musdb18 -o dataset/musdb18
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
- either "image" or "1d_signal"

To this you can add a "config" entry giving overrides to the standard configuration for the model. See models/trainer/standard_configurations to see all possible options. For example:

```
{
    "pathDB": PATH_TO_YOUR_DATASET,
    "dbType": DATASET_TYPE,
    "config": {"dataType": "vector",
               "baseLearningRate": 0.1,
               "miniBatchSize": 22}
}
```

Will override the learning rate and the mini-batch-size. Please note that if you specify a - -baseLearningRate option in your command line, the command line will prevail. Depending on how you work you might prefer to have specific configuration files for each run or only rely on one configuration file and input your training parameters via the command line.

### dbType vs dataType

Often in NN training, we may require target (e.g. data inputs) and generated/processed data to be of different types. For example, we may require the model to generate vector information while comparing the generated vector information in the rasterized domain.

- dbType refers to the data type in the training data (which determines the type of dataloader to use)
- dataType refers to the data type of the output of the network (which determines the types of output layers to use)

For example, given the previous example, dbType = "image" while dataType = "vector"

## TODO List:

- Implement AEGAN architecture
- Implement GAN to replace rasterizer
- Implement UNet
- Implement GE for SS
- Restructure Code for easy access
- Restructure datasets.py
- Build requirements.txt

## CREDITS

The following sources have contributed greatly to the development of this repository:

- GAN Architecture, base code: https://github.com/facebookresearch/pytorch_GAN_zoo
- Differentiable Vector Graphics: https://github.com/BachiLi/diffvg
- VAE References: https://github.com/sksq96/pytorch-vae, https://github.com/atinghosh/VAE-pytorch
- Generative Encoder for Image Processing: https://www.researchgate.net/publication/342464064_Digital_Rock_Image_Inpainting_using_GANs
