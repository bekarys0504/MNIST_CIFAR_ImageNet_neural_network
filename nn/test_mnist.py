import gzip
import numpy as np
import matplotlib.pyplot as plt
from mnist import Net
import torch.nn as nn
import torch
from PIL import Image, ImageFilter
import os 

print(os.getcwd())
f = gzip.open('/home/beka/data/MNIST/raw/t10k-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 10

f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
#print(data[0].shape)
data = data.reshape(num_images, image_size, image_size, 1)
print(data.shape)
image = np.asarray(data).squeeze()
image = image.reshape(num_images, 1, image_size, image_size)
image = torch.from_numpy(image)
#load weights
model = Net()
model.load_state_dict(torch.load("mnist_5fc.pt"))
model.eval()

output = model(image)
print(output)
pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability


# Plot images in columns and indicating predicted number below
fig,axes = plt.subplots(nrows = 1, ncols = num_images, figsize=(30,10))

for ax in axes.flatten():
    ax.axis('off')

for i in range(0, num_images):
    imgplot = image[i].reshape(image_size, image_size)
    axes[i].imshow(imgplot, cmap='gray')   
    axes[i].title.set_text(pred.numpy()[i][0])

plt.show()