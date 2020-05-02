import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import metrics
import torch.optim as optim
from algorithms.vgg import VGG
from algorithms.lenet import LeNet
from algorithms.alexnet import AlexNet
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, datasets
import numpy as np

epoches = 5
start_epoch = 1
batchsize = 10
learning_rate = 0.001
actual = []
prediction = []

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
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


model = AlexNet()
# model = LeNet()
# model = VGG()
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
# print(' '.join('%5s' % classes[labels[j]] for j in range(10)))

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr= learning_rate)
optimizer = optim.SGD(model.parameters(), lr= learning_rate, momentum = 0.9)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)


def train(model, device, train_loader, optimizer, epoch):
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
            #total += target.size(0)
            actual.extend(target.view_as(predicted))
            prediction.extend(predicted)
            correct += predicted.eq(target).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


for epoch in range(start_epoch, epoches + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()

print('Confusion matrix:')
print(metrics.confusion_matrix(actual, prediction))
print(metrics.classification_report(actual, prediction, digits=3))



   # if save_model:
   #     torch.save(model.state_dict(), "./results/cifar10_cnn.pt")
# # plot loss during training
# plt.subplot(211)
# plt.title('Loss')
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# # plot accuracy during training
# plt.subplot(212)
# plt.title('Accuracy')
# plt.plot(history.history['accuracy'], label='train')
# plt.plot(history.history['val_accuracy'], label='test')
# plt.legend()
# plt.show()



# def imsave(img):
#     npimg = img.numpy()
#     npimg = (np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8)
#     im = Image.fromarray(npimg)
#     im.save("./results/your_file.jpeg")
#
# def train_cnn(log_interval, model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward();
#         optimizer.step()
#         if batch_idx % log_interval == 0:
#             print ('Train Epoch: {} [{}/{} (:.0f)%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx*len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader)
#                                                                        , loss.item()))
#
# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             loss = criterion(output, target)
#             test_loss += loss.item()
#             _, predicted = output.max(1)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#     test_loss /= len(test_loader.dataset())
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
#
# criterion = nn.CrossEntropyLoss()
# def main():
#     epoches = 14
#     gamma = 0.7
#     log_interval = 10
#     torch.manual_seed(1)
#     save_model = True
#
#     RNN = False # Check
#     N_STEPS = 28
#     N_INPUTS = 28
#     N_NEURONS = 150
#     N_OUTPUTS = 10
#
#     device = torch.device("cpu") # Do not have cuda on laptop
#
#     #Torchvision
#
#     kwargs = {}
#     train_loader = torch.utils.data.DataLoader(
#         datasets.CIFAR10('../data', train=True, download=True,
#                          transform=transforms.Compose([
#                              transforms.ToTensor()])), batch_size=64, shuffle=True, **kwargs)
#
#     test_loader = torch.utils.data.DataLoader(
#         datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
#                              transforms.ToTensor()])), batch_size=1000, shuffle=True, **kwargs)
#
#     dataiter = iter(train_loader)
#     images, labels = dataiter.next()
#     #check img = torchvision.utils.make_grid(images)
#     #check imsave(img)
#
#     model = LeNet().to(device)
#     optimizer = optim.Adam(model.parameters(), lr= 0.001)
#     #scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
#
#     for epoch in range(1, epoches + 1):
#         train_cnn(log_interval, model, device, train_loader, optimizer, epoch)
#         test(model, device, test_loader)
#         #scheduler.step()
#
#     if save_model:
#         torch.save(model.state_dict, "./results/cifar_cnn.pt")
#
# if __name__ == '__main__':
#     main()