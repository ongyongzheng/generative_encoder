import os
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
import torch.utils.data as data

from config import Config # TODO: may not be needed
from utils import get_folder_dir

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)
    return images

class ImageFolder(data.Dataset):
    def __init__(self, root,
        transform=None,
        target_transform=None,
        loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root))
        print("Found {} images in subfolders of: {}".format(len(imgs), root))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img

    def __len__(self):
        return len(self.imgs)

def make_loader(root, batch_size, mode, img_size, num_workers=2, shuffle=True):
    folder = get_folder_dir(mode)[0]
    transform = transforms.Compose([
        #transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # load train data
    train_data = ImageFolder(
        os.path.join(root, folder, 'img_resize_' + str(img_size), 'train'),
        transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=shuffle, num_workers=int(num_workers), drop_last=True)

    # load test data
    test_data = ImageFolder(
        os.path.join(root, folder, 'img_resize_' + str(img_size), 'test'),
        transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=int(num_workers))
    return train_loader, test_loader