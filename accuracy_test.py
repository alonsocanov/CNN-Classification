import torch
import functions as func
import torchvision
import torch.optim as optim
import my_model as m
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

path_my_idx = 'label_idx.csv'

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img_resize = 256
img_crop = 224
transform = transforms.Compose([
    transforms.Resize(img_resize),
    transforms.CenterCrop(img_crop),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])


# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--path', type=str, default='../data/person_detection/')
argparser.add_argument('--labels', default=np.array(['car_side', 'cellphone', 'person']))
argparser.add_argument('--n_epochs', type=int, default=2)
argparser.add_argument('--batch_size', type=int, default=16)
argparser.add_argument('--criterion', default=nn.CrossEntropyLoss())
argparser.add_argument('--momentum', type=float, default=0.9)
argparser.add_argument('--transform', default=transform)
argparser.add_argument('--learning_rate', type=float, default=0.001)
argparser.add_argument('--save_model', type=bool, default=False)
argparser.add_argument('--feature_extract', type=bool, default=True)
args = argparser.parse_args()
print(args)

data_set = torchvision.datasets.ImageFolder(root=args.path, transform=args.transform)
# print(data_set.targets)
print(data_set.class_to_idx)
train_size = len(data_set) // 2
test_size = len(data_set) - train_size
data_train, data_test = random_split(data_set, [train_size, test_size])
train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=False, sampler=None, num_workers=0)
test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, sampler=None, num_workers=0)

print('Train sample: ', train_size)
print('Test sample: ', test_size)

# my model
# checkpoint = torch.load('person_detection.pth')
# fine tuned model
checkpoint = torch.load('person_detection_ft.pth')

model = checkpoint['model']
model.eval()

func.accuracy_test(model, test_loader)


img = iter(test_loader)
images, lab = img.next()

# print images
func.imshow(torchvision.utils.make_grid(images), mean, std, args.labels[lab])


outputs = model(images)
_, predicted = torch.max(outputs, 1)
print('Predicted :', args.labels[predicted])



class_correct = torch.zeros(len(args.labels))
class_total = torch.zeros(len(args.labels))
with torch.no_grad():
    for data in test_loader:
        images, lab = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == lab)
        for i in range(len(c)):
            label = lab[i]
            class_correct[label] += c[i]
            class_total[label] += 1
print(class_total)
for i in range(len(args.labels)):
    print('Accuracy of %5s : %2d %%' % (args.labels[i], 100 * class_correct[i] / class_total[i]))

