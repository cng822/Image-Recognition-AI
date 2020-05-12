import torch
import torch.nn as nn

# Defined the different output layers
VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
# Number of classes in CIFAR-10
NUM_CLASSES = 10

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.convs = self.make_layers(VGG16)  # convolution function, passed through for feature extraction
        self.fc = nn.Linear(512, NUM_CLASSES) # classification layer

    def forward(self,x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)   # flattens the image into a column to prepare for classification
        return self.fc(x)


    def make_layers(self, VGG16):
        layers = []                # array to store the layers after convolution
        in_channels = 3            # first in channels = 3 as image as input has RBG
        for i in VGG16:            # loops through all values in the VGG-16 array
            if i == 'M':
                layers += [nn.MaxPool2d(2)]    # if the current value in VGG-16 = 'M' down sample and find the most prominent feature
            else:
                layers += [nn.Conv2d(in_channels, i, 3, padding = 1), # in channels = 3, out channels = current vgg-16 value
                           nn.BatchNorm2d(i),
                           nn.ReLU(inplace = True)] # activation function
                in_channels = i         # current vgg-16 channel will become the new input
        return nn.Sequential(*layers)   # returns vgg-16 after convolution