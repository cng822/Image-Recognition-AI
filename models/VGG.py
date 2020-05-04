import torch
import torch.nn as nn
import torch.nn.functional as F

VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
NUM_CLASSES = 10
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.convs = self.make_layers(VGG16)
        self.classifier = nn.Linear(512, NUM_CLASSES)

    def forward(self,x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


    def make_layers(self, VGG16):
        layers = []
        in_channels = 3
        for i in VGG16:
            if i == 'M':
                layers += [nn.MaxPool2d(2)]
            else:
                layers += [nn.Conv2d(in_channels, i, 3, padding = 1),
                           nn.BatchNorm2d(i),
                           nn.ReLU(inplace = True)]
                in_channels = i

        layers += [nn.AvgPool2d(1,1)]
        return nn.Sequential(*layers)
