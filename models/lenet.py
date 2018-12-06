import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=43, input_channels=3):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)
        if 1 == num_classes: 
            # compatible with nn.BCELoss
            self.softmax = nn.Sigmoid()
        else:    
            # compatible with nn.CrossEntropyLoss
            self.softmax = nn.LogSoftmax(dim=1)        


    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        out = self.softmax(out)
        return out

def lenet5( **kwargs):
    model = LeNet(**kwargs)
    return model


def lenet(model_name, num_classes, input_channels, pretrained=False):
    return{
        'lenet5': lenet5(num_classes=num_classes, input_channels=input_channels),
    }[model_name]