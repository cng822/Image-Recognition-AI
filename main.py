from __future__ import print_function
import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torch.nn.functional as F
from PIL import Image
from cnn import Net

def imsave(img):
    npimg = img.numpy()
    npimg = (np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8)
    im = Image.fromarray(npimg)
    im.save("./results/your_file.jpeg")

def train_cnn(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward(); optimizer.step()
        if batch_idx % log_interval == 0:
            print ('Train Epoch: {} [{}/{} (:.0f)%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader)
                                                                       , loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = torch.squeeze(data)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset())
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

def main():
    epoches = 14
    gamma = 0.7
    log_interval = 10
    torch.manual_seed(1)
    save_model = True

    RNN = False # Check
    N_STEPS = 28
    N_INPUTS = 28
    N_NEURONS = 150
    N_OUTPUTS = 10

    device = torch.device("cpu") # Do not have cuda on laptop

    #Torchvision

    kwargs = {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor()])), batch_size=64, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                             transforms.ToTensor()])), batch_size=1000, shuffle=True, **kwargs)

    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    #check img = torchvision.utils.make_grid(images)
    #check imsave(img)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr= 0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(1, epoches + 1):
        train_cnn(log_interval, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if save_model:
        torch.save(model.state_dict, "./results/cifar_cnn.pt")

if __name__ == '__main__':
    main()