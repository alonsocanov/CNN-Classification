import os
import torch
import pandas as pd
import torchvision.transforms as transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import variables as var
import glob
from PIL import Image


class Image_data(Dataset):
    def __init__(self, path, classes, transform=None):
        self.path = path
        self.classes = classes
        self.transform = transform
        self.img_names = []
        self.labels = []
        n = 0
        for i in classes:
            for im in glob.glob(self.path + i + '/*.jpg', recursive=True):
                self.img_names.append(im)
                self.labels.append(n)
            n += 1

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        img = Image.open(self.img_names[item]).convert('RGB')
        if transform:
            img = var.transform(img)
        img_data = {'image': img, 'class': self.labels[item]}
        return img_data


path = '../data/Object_Categories/'
classes = ['car_side', 'cellphone', 'person']
data = Image_data(path, classes, var.transform)
print(data[1]['image'])
plt.imshow(data[1]['image'].permute(1, 2, 0))
plt.show()
print(data[1]['class'])
loader = DataLoader(data, batch_size=2, shuffle=False, sampler=None, num_workers=0)