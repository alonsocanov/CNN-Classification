import torch.nn as nn
import torch.nn.functional as F

# my CNN model
# takes a 224x224 image size


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(in_features=32 * 25 * 25, out_features=250)
        self.fc2 = nn.Linear(250, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 10)
        self.fc5 = nn.Linear(10, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 25 * 25)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
