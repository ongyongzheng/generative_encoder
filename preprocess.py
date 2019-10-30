import os
import matplotlib.pyplot as plt
from scipy.misc import imresize

from config import Config

def get_folder_dir(mode):
    if mode == 'celeba':
        return 'CelebA'

if __name__ == "__main__":
    config = Config()

    resize_size = config.img_size
    folder_name = get_folder_dir(config.mode)

    folder_dir = os.path.join(os.path.join(config.dataset_path, folder_name), 'img_resize_' + str(resize_size))
    if len(os.listdir(folder_dir)) == 0:
        print('found empty save directory, preparing to resize images')



