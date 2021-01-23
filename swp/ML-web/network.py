import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
         # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 6)
    
    #def forward(self, x):
        #x = F.relu(self.conv1(x))
        #x = F.max_pool2d(x, 2, 2)
        #x = self.conv1_bn(x)
        #x = F.relu(self.conv2(x))
        #x = F.max_pool2d(x, 2, 2)
        #x = x.view(-1, 2048)
        #x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        #x = self.fc2(x)
        #x = x.view(-1, 1, 512)
        #x = self.bn(x)
        #x = x.view(-1, 512)
        #x = self.fc3(x)
        #x = self.fc4(x)

        #return x


    def forward(self, xb):
        return torch.sigmoid(self.network(xb))