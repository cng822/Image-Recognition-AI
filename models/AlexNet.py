import torch
import torch.nn as nn
import torch.nn.functional as F

#Constant
NUM_CLASSES = 10

class AlexNet (nn.Module) :

    def __init__(self):
        super(AlexNet, self).__init__()
        # input channels = 3 (rgb colours) out channels = 64; feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)     
        
        # input channels = 64 (last conv's out channel), out channels = 192
        self.conv2 = nn.Conv2d(64, 192, kernel_size = 3, padding = 1)
        
        # input channels = 192 (last conv's out channel), out channels = 384
        self.conv3 = nn.Conv2d(192, 384, kernel_size= 3, padding = 1)
        
        # input channels = 384 (last conv's out channel), out channels = 256
        self.conv4 = nn.Conv2d(384, 256, kernel_size = 3, padding = 1)
        
        # input channels = 256 (last conv's out channel), out channels = 256
        self.conv5 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        
         # drop out to filter the noise out
        self.dropout1 = nn.Dropout2d(0.25)     
        self.dropout2 = nn.Dropout2d(0.5)

        # fully connected layers - where classification happens
        self.fc1 = nn.Linear(256 * 4 * 4, 4096) 
        self.fc2 = nn.Linear(4096, 4096)
        
        # dense layer 
        self.fc3 = nn.Linear(4096, NUM_CLASSES) 

    def forward(self, x):
        x = F.relu(self.conv1(x))      # activation layer is applied to the convolution layer
        x = F.max_pool2d(x, 2)         # max pool occurs, finds the maximum value ( prominent feature)
        x = F.relu(self.conv2(x))       
        x = F.max_pool2d(x,2)           
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x,2)          # max pool occurs for the 3rd time after activation layer ReLU is applied to 3 more convolutional layers

        x = x.view(x.size(0), -1)      # flattens the layer into a column

        x = self.dropout1(x)           # filters out any noise before it passes through the first classification layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)                
        return x                    



