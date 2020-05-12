import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import metrics
import torch.optim as optim
from models.VGG16 import VGG16
from models.ResNet import ResNet, BasicBlock
from models.AlexNet import AlexNet
from models.SENet import SENet, BasicBlock 
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, datasets
import numpy as np

# CONSTANTS
epoches = 10
start_epoch = 1
learning_rate = 0.001

# Arrays to store values for our plots and confusion matrix
actual = []
prediction = []
loss_graph = []
accuracy_graph = []
accuracy_train = []
loss_test = []

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Imports the training dataset
train_CIFAR10 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform= transform )

# Imports the testing dataset
test_CIFAR10 = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform= transform)
# loads training dataset
train_loader = torch.utils.data.DataLoader(train_CIFAR10, batch_size = 32, shuffle = True)

# loads testing dataset
test_loader = torch.utils.data.DataLoader(test_CIFAR10, batch_size = 64, shuffle = True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# check if CUDA is available, if yes, it will use gpu if not then it will use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if device else {}

# Selecting which model to use
print("Please enter the number of one of the following models: \n AlexNet : 1 \n ResNet50 : 2 \n SENet : 3 \n VGG16 : 4")
val = input("Enter here: ")
found = 0

# if value is found then model is selected and while loop exits
while found != 1:
    if val == "1":
        model = AlexNet()
        found = 1
    elif val == "2":
        model = ResNet(BasicBlock, [3, 4, 6, 3])
        found = 1
    elif val == "3":
        model = SENet(BasicBlock, [2, 2, 2, 2])
        found = 1
    elif val == "4":
        model = VGG()
        found = 1
    else:
        val = input("Please enter again: ")


model = model.to(device)

# showing the image function
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))
    # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(10)))

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer function
optimizer = optim.Adam(model.parameters(), lr= learning_rate)

# Step function
scheduler = StepLR(optimizer, step_size=1, gamma= 0.7)

# Training phase
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    loss_tol = 0
    correct = 0
	
    # loops through the training set
    for batch_no, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() #resets
        output = model(data)
        loss = criterion(output, target)  
        loss.backward() # computes gradients
        optimizer.step()
        loss_tol += loss.item() 	
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item() 	# if correct, increment 1
        if batch_no % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_no * len(data), len(train_loader.dataset),
                100. * batch_no / len(train_loader), loss.item()))
    loss_graph.append(loss_tol / len(train_loader.dataset))		# prints out which epoch and the loss per batch
    accuracy_train.append(correct / len(train_loader.dataset))		

# Test phase
def test(model, device, test_loader):
    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = output.max(1)			
            actual.extend(target.view_as(predicted))		# stores actual image
            prediction.extend(predicted)			# stores predicted output
            correct += predicted.eq(target).sum().item()
        tol_loss = test_loss
        test_loss /= len(test_loader.dataset)
        accuracy_graph.append(correct / len(test_loader.dataset))
        loss_test.append(tol_loss / len(test_loader.dataset))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset))) 	# prints out average accuracy of that epoch


# Loops through the amount of epoch
for epoch in range(start_epoch, epoches + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()

print('Confusion matrix:')

# Prints a confusion matrix where we see what the neural network predicted vs what the image class actual
print(metrics.confusion_matrix(actual, prediction))

# Prints the precision, recall and f1 score
print(metrics.classification_report(actual, prediction, digits=3))


# Plots a loss graph
plt.subplot(211)
plt.plot(loss_graph, label="training")
plt.plot(loss_test, label="testing")
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.legend(loc="upper right")

# Plots an accuracy graph
plt.subplot(212)
plt.plot(accuracy_train, label="training")
plt.plot(accuracy_graph, label="testing")
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy(%)')
plt.legend(loc="upper right")
plt.show()

# Figure displays two plots - Top: Loss Bottom: Accuracy
