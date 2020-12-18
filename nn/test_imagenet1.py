import numpy as np
import matplotlib.pyplot as plt
from imagenet import AlexNet
#from cifar import Net1
import torch.nn as nn
import torch
from PIL import Image, ImageFilter
import os
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import torchvision

def imshow(img):
        img = img*0.224 + 0.456     # unnormalize
        npimg = img.numpy()
        print(npimg.shape)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

#Define initial parameters
classes = ('fish', 'dog', 'radio', 'saw', 'house', 'clarnet', 'truck', 'gas station', 'ball', 'parachute')
NUM_IMAGES = 10
TEST_DIR = './test'
num_classes = 10

#normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_transform = transforms.Compose([
    transforms.CenterCrop(227),
    transforms.ToTensor(),
    normalization
])
dataset_test = datasets.ImageFolder(TEST_DIR, transform=test_transform)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=NUM_IMAGES, shuffle=True)

dataiter = iter(test_loader)
images, labels = dataiter.next()

net = AlexNet(num_classes)
net.load_state_dict(torch.load("imagenet.pt"))
outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                            for j in range(NUM_IMAGES)))


imshow(torchvision.utils.make_grid(images))