import numpy as np
import matplotlib.pyplot as plt
from imagenet import AlexNet
#from cifar import Net1
import torch.nn as nn
import torch
from PIL import Image, ImageFilter
import os

dictionary = {0: 'Fish', 1: 'Dog', 2: 'Radio', 3: 'Saw', 4: 'House', 5: 'Sax', 6: 'Car', 7: 'Gas', 8: 'Ball', 9: 'Parachute'}
count = 0
for img_name in os.listdir('test/1'):
    #print(img_name)
    im = Image.open('test/1/' + img_name)
    width, height = im.size   # Get dimensions
    
    #Crop image to the size 227x227
    left = width/4
    top = height/4
    right = 227+width/4
    bottom = 227+ height/4
            #print(x.shape)
    im = im.crop((left, top, right, bottom))
    #print(im.size)
    # im.show()
    image_size = 227
    num_classes = 10
    image = np.array(im).astype(np.float32)
    image = (image - 0.485) / 0.224
    print(image.shape)
    image = image.reshape(1, 3, image_size, image_size)
    #image = np.asarray(data)
    # print(image.shape)
    #image = image.reshape(1, 3, image_size, image_size)
    #image1 = image.reshape(image_size, image_size, 3)
    #PILimage1 = Image.fromarray(image1.astype('uint8'), 'RGB')

    #PILimage1.show()
    image = torch.from_numpy(image)
    #load weights
    model = AlexNet(num_classes)
    model.load_state_dict(torch.load("imagenet.pt"))
    model.eval()
    output = model(image)
    # print(output)
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    print(dictionary[pred.numpy()[0][0]])
    if dictionary[pred.numpy()[0][0]] == 'Dog':
        print("Correct")
        count += 1

print(count, "out of", 395)
#infer each image class model(img)
#print class

#fig,axes = plt.subplots(nrows = 1, ncols = num_images, figsize=(30,10))

#for ax in axes.flatten():
#    ax.axis('off')

#for i in range(0, num_images):
    #print('Printing shape')
    #print(image.shape)
    #imgplot = image[i].reshape(3, image_size, image_size)
    #axes[i].imshow(imgplot, cmap='gray')   
    #axes[i].title.set_text(pred.numpy()[i][0])

#plt.show()