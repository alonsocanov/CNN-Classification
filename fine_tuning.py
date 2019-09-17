from torchvision.models import alexnet
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch
import torch.optim as optim
import numpy as np
import my_model as m
import argparse
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision
import functions as func

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img_resize = 256
img_crop = 224
transform = transforms.Compose([
    transforms.Resize(img_resize),
    transforms.CenterCrop(img_crop),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('GPU: ', device)

argparser = argparse.ArgumentParser()
argparser.add_argument('--path', type=str, default='../data/person_detection/')
argparser.add_argument('--labels', default=np.array(['car_side', 'cellphone', 'person']))
argparser.add_argument('--n_epochs', type=int, default=2)
argparser.add_argument('--batch_size', type=int, default=16)
argparser.add_argument('--criterion', default=nn.CrossEntropyLoss())
argparser.add_argument('--momentum', type=float, default=0.9)
argparser.add_argument('--transform', default=transform)
argparser.add_argument('--print_every', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=0.001)
argparser.add_argument('--save_model', type=bool, default=False)
argparser.add_argument('--feature_extract', type=bool, default=True)
args = argparser.parse_args()
print(args)

data_set = torchvision.datasets.ImageFolder(root=args.path, transform=args.transform)
classes = func.find_classes(args.path)
# print(data_set.targets)
print(data_set.class_to_idx)
train_size = len(data_set) // 2
test_size = len(data_set) - train_size
data_train, data_test = random_split(data_set, [train_size, test_size])
train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=False, sampler=None, num_workers=0)
test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, sampler=None, num_workers=0)

print('Train sample: ', train_size)
print('Test sample: ', test_size)


model = alexnet(pretrained=True)
model = func.set_parameter_requires_grad(model, args.feature_extract)
model.classifier[6] = nn.Linear(in_features=4096, out_features=len(args.labels))

print("Params to learn:")
params_to_update = func.to_grad_param(model, args.feature_extract)

optimizer = optim.SGD(params_to_update, lr=args.learning_rate, momentum=args.momentum)

def train(train_load):
    running_loss = 0.0
    for i, data in enumerate(train_load, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, lab = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = args.criterion(outputs, lab)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (i+1) % args.batch_size == 0:
            print('[%d, %5d] loss: %.8f' % (epoch + 1, i + 1, running_loss / args.batch_size))
            running_loss = 0.0

try:
    loss = 0
    # train model
    for epoch in range(args.n_epochs):  
        train(train_loader)
    print('Finished Training')
    if args.save_model:
        print('Saving model')
        state = {
            'model': model,
            'epoch': args.n_epochs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss
        }
        torch.save(state, 'person_detection_ft.pth')
except KeyboardInterrupt:
    print('\nTraining interrupted')
    if args.save_model:
        print('Saving model')
        state = {
            'model': model,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss
        }
        torch.save(state, 'person_detection_ft.pth')

