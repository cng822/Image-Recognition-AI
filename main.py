import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from algorithms.alexnet import AlexNet
from algorithms.vgg import VGG
from algorithms.lenet import LeNet
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np

epoches = 5
start_epoch = 1
batchsize = 10
learning_rate = 0.001
transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_CIFAR10 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform= transform )

test_CIFAR10 = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform= transform)

train_loader = torch.utils.data.DataLoader(train_CIFAR10, batch_size = batchsize, shuffle = True)# batch_size to change

test_loader = torch.utils.data.DataLoader(test_CIFAR10, batch_size = batchsize, shuffle = True)# batch_size to change

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if device else {}
#model = AlexNet()
model = LeNet()
model = VGG()
model = model.to(device)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
    # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate)

def train(epoch):
    model.train()
    for batch_no, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_no % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_no * len(data), len(train_loader.dataset),
                100. * batch_no / len(train_loader), loss.item()))


def test(epoch):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        # # Save checkpoint.
        # acc = 100. * correct / total
        # if acc > best_acc:
        #     print('Saving..')
        #     state = {
        #         'net': net.state_dict(),
        #         'acc': acc,
        #         'epoch': epoch,
        #     }
        #     torch.save(state, './checkpoint/ckpt.pth')
        #     best_acc = acc
#
# def adjust_learning_rate(optimizer, epoch):
#     global state
#     if epoch in args.schedule:
#         state['lr'] *= args.gamma
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = state['lr']
#
for epoch in range(start_epoch, epoches + 1):
   train(epoch)
   test(epoch)
