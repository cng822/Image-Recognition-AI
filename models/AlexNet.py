import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 10
class AlexNet (nn.Module) :

    def __init__(self):
        super(AlexNet, self).__init__()

        # self.cnnlayers = nn.Sequential(
            #defining first convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            # self.ReLU1 = nn.ReLU(inplace = True),
            # self.maxpool1 = nn.MaxPool2d(kernel_size = 2),

            #defining second convolutional layer
        self.conv2 = nn.Conv2d(64, 192, kernel_size = 3, padding = 1)
            # self.ReLU2 = nn.ReLU(inplace= True),
            # self.maxpool2 = nn.MaxPool2d(kernel_size = 2),

            #defining third, fourth, last convolutional layer
        self.conv3 = nn.Conv2d(192, 384, kernel_size= 3, padding = 1)
            # self.RELU3 = nn.ReLU(inplace= True),

        self.conv4 = nn.Conv2d(384, 256, kernel_size = 3, padding = 1)
            # self.ReLU4 = nn.ReLU(inplace= True)

        self.conv5 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
            # self.ReLU5 = nn.ReLU(inplace= True)
            # self.maxpool3 = nn.MaxPool2d(kernel_size = 2)

        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, NUM_CLASSES)

       #)

        # self.fclayers = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Linear(4096, NUM_CLASSES),
        # )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x,2)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim = 1)
        return x



       #  x = self.cnnlayers(x)
       #  x = x.view(x.size(0), -1)
       #  x = self.fclayers(x)
       # # x = F.log_softmax(x, dim=3)
       #  return x



x = torch.randn(64,3,32,32)
model = AlexNet()
print(model(x).shape)