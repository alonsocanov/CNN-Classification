import variables as var
from torchvision.models import alexnet
import torch.nn as nn
import torch.optim as optim
import torch
import functions as func


feature_extract = True
save_model = False
# data_set = var.data_set
# classes = var.classes
# input_size = var.img_crop
# train_size = var.train_size
# test_size = var.test_size

train_loader = var.train_loader
test_loader = var.test_loader


model = alexnet(pretrained=True)
model = func.set_parameter_requires_grad(model, feature_extract)
model.classifier[6] = nn.Linear(in_features=4096, out_features=var.num_classes)

print("Params to learn:")
params_to_update = func.to_grad_param(model, feature_extract)

optimizer = optim.SGD(params_to_update, lr=var.lr, momentum=var.momentum)

loss = 0
# train model
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
            print('[%d, %5d] loss: %.8f' % (epoch + 1, i + 1, running_loss / var.batch_size))
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
if save_model:
    torch.save(state, 'person_detection_ft.pth')
