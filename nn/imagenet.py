from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

num_classes = 10

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        #First convolutional layer 3x3, s = 4, p = 0
        self.conv1 = nn.Conv2d(3, 96, 11, 4)
        #Max pooling 3x3, s = 2
        self.pool = nn.MaxPool2d(3, 2)
        #Local response normalization
        self.lrn = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        #Second convolutional layer 5x5, s = 1, p = 2
        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)
        #Third convolutional layer 3x3, s = 1, p = 1
        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        #Fourth convolutional layer 3x3, s = 1, p = 1
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        #Fourth convolutional layer 3x3, s = 1, p = 1
        self.conv5 = nn.Conv2d(384, 256, 3, 1, 1)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.conv1(x))
        x = self.lrn(x)
        x = self.pool(x)
        #print(x.shape)
        x = self.pool(self.lrn(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        #print(x.shape)
        x = F.relu(self.conv4(x))
        #print(x.shape)
        x = self.pool(F.relu(self.conv5(x)))
        #print(x.shape)
        x = x.view(-1, 256 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                                                   
        return x


def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


def train(log_interval, model, device, train_loader, optimizer, epoch):

    criterion = nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #print(target)
            #print(output)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            _, pred = torch.max(output, 1)
            correct += (pred == target).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    classes = ('fish', 'dog', 'radio', 'saw',
            'house', 'clarnet', 'truck', 'gas station', 'ball', 'parachute')
    use_cuda = torch.cuda.is_available()

    #torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalization
    ])
    test_transform = transforms.Compose([
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        normalization
    ])
    dataset_train = datasets.ImageFolder('/home/beka/imagenette2-320/train', transform=train_transform)
    dataset_test = datasets.ImageFolder('/home/beka/imagenette2-320/val', transform=test_transform)
    #print(dataset_test)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=True)
    #print(test_loader)
    model = AlexNet(num_classes = num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(1, 21):
        train(10, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    torch.save(model.state_dict(), "imagenet.pt")

if __name__ == '__main__':
    main()