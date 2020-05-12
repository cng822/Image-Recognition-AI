import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # applies the first convolution
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # applies the second convolution
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # contains all modules passed into it in the correct sequence
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        # SE layers - classification
        self.fc1 = nn.Conv2d(planes, planes // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes // 16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # apply the activation layer to the convolution layer
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))  # apply average 2d pooling over the multiple layers
        w = F.relu(self.fc1(w))  # apply the activation layer
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += self.shortcut(x)  # add all the modules
        out = F.relu(out) # apply the activation layer to out
        return out


class SENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SENet, self).__init__()
        self.in_planes = 64
        # input channels = 3 for the three colours, output channels = 64,
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # make the first layer with 64 planes
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # make the second layer with 128 planes
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # make the third layer with 256 planes
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # make the fourth layer with 512 planes
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # transform
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []  # create an empty array for the layers to be stored in after convolution
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # apply the activation layer to the convolution layer
        out = self.layer1(out)  # makes the first layer of the image
        out = self.layer2(out)  # makes the second layer of the image
        out = self.layer3(out)  # makes the third layer of the image
        out = self.layer4(out)  # makes the fourth layer of the image
        # applies a 2d convolution over the four layer image
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # applies a linear transformation to make a fully connected linear layer to output
        out = self.linear(out)
        return out
