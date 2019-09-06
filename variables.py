import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision
import functions as func

path = '../data/person_detection/'
path_my_idx = 'label_idx.csv'
loc = 'Index of Classes'
labels = np.array(['car_side', 'cellphone', 'person'])
num_classes = len(labels)
print('Data: ', path)
print('Labels: ', labels)


batch_size = 5
epochs = 1
lr = .001
criterion = nn.CrossEntropyLoss()
momentum = 0.9
print('Batch size: ', batch_size)
print('Epoch: ', epochs)
print('Learning rate: ', lr)
print('Criterion: ', criterion)
print('Momentum: ', momentum)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
print('Mean: ', mean)
print('Standard Deviation: ', std)

img_resize = 256
img_crop = 224

transform = transforms.Compose([
    transforms.Resize(img_resize),
    transforms.CenterCrop(img_crop),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])

print('Transform: ', transform)
data_set = torchvision.datasets.ImageFolder(root=path, transform=transform)
classes = func.find_classes(path)
# print(data_set.targets)
print(data_set.class_to_idx)
train_size = len(data_set) // 2
test_size = len(data_set) - train_size
data_train, data_test = random_split(data_set, [train_size, test_size])
train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=False, sampler=None, num_workers=0)
test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False, sampler=None, num_workers=0)
