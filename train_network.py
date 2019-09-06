import torch
import torch.optim as optim
import functions as func
import numpy as np
import torchvision
import variables as var
import my_model as m

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('GPU: ', device)


path = '../data/person_detection/'
path_my_idx = 'label_idx.csv'
loc = 'Index of Classes'
labels = np.array(['car_side', 'cellphone', 'person'])
print('Data: ', path)
print('Labels: ', labels)

transform = var.transform

data_set = var.data_set
classes = func.find_classes(path)
# print(data_set.targets)
print(data_set.class_to_idx)
train_size = len(data_set)//2
test_size = len(data_set) - train_size
print('Train sample: ', train_size)
print('Test sample: ', test_size)

train_loader = var.train_loader
test_loader = var.test_loader

img = iter(train_loader)
images, lab = next(img)
func.imshow(torchvision.utils.make_grid(images), var.mean, var.std, labels[lab])

model = m.Net()
optimizer = optim.SGD(model.parameters(), lr=var.lr, momentum=var.momentum)

for epoch in range(var.epochs):  # loop over the data set multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, lab = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = var.criterion(outputs, lab)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i-1 % var.batch_size == 0:
            print('Epoch: %d, batch: %5d -> loss: %.8f' % (epoch + 1, i + 1, running_loss / var.batch_size))
            running_loss = 0.0

print('Finished Training')
print('Saving model')

state = {
    'model': model,
    'epoch': var.epochs,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'loss': loss
}

torch.save(state, 'person_detection.pth')
