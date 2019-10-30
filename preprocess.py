import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from progress.bar import Bar

from config import Config
from utils import get_folder_dir

def preprocess(img_list, save_dir, img_dir, size):
    print("saving images to {}".format(save_dir))
    bar = Bar('Resizing', max=len(img_list))
    for img in img_list:
        image = Image.open(os.path.join(img_dir, img))
        new_image = image.resize((size, size), Image.ANTIALIAS)
        new_image.save(os.path.join(save_dir, img))
        bar.next()
    bar.finish()


if __name__ == "__main__":
    config = Config()

    resize_size = config.img_size
    folder_name = os.path.join(config.dataset_path, get_folder_dir(config.mode)[0])

    folder_dir = os.path.join(folder_name, 'img_resize_' + str(resize_size))
    if len(os.listdir(os.path.join(folder_dir, 'train'))) == 0:
        print('found empty save directory, preparing to resize images')
        # load images
        load_dir = os.path.join(folder_name, get_folder_dir(config.mode)[1])
        img_list = os.listdir(load_dir)
        print('{} images found: {}'.format(config.mode, len(img_list)))
        train_list, test_list = train_test_split(img_list, test_size=config.test_size)

        preprocess(train_list, os.path.join(folder_dir, 'train'), load_dir, resize_size)
        preprocess(test_list, os.path.join(folder_dir, 'test'), load_dir, resize_size)
    else:
        print('save directory already contains files, please clear and try again')





