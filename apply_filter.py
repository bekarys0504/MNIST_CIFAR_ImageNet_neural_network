from PIL import Image, ImageFilter 
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
import numpy as np
from scipy.signal import correlate

def apply_filter(img, padding, filter1):
    img = np.array(img)    
    row = 0
    col = 0
    
    if padding == 0:
        newimg = np.zeros(shape=(len(img[:, 0])-2,len(img[0, :])-2))
    else:
        np.pad(img, 1, mode='constant')
        newimg = np.zeros(shape=(len(img[:, 0]),len(img[0, :])))
    
    for i in range(0, len(img[:, 0])-2):
        row = 0
        for j in range(0, len(img[0, :])-2):
            newimg[col, row] = np.sum(np.multiply(img[i:i+3, j:j+3], filter1))
            row = row+1
        col = col+1

    print(newimg.shape)
    arr = correlate(img, filter1)
    arr = arr.astype(np.float64)
    im = Image.fromarray(newimg)
    im.show()

im = Image.open("output1.jpg")
# Converting the image to greyscale, as Sobel Operator requires 
# input image to be of mode Greyscale (L) 
img = im.convert("L")

# Calculating Edges using the passed laplican Kernel 
sobelVer = img.filter(ImageFilter.Kernel((3, 3), (1, 0, -1, 2, 0, -2, 1, 0, -1), 1, 0)) 
sobelHor = img.filter(ImageFilter.Kernel((3, 3), (1, 2, 1, 0, 0, 0, -1, -2, -1), 1, 0))

#gaussBlur = img.filter(ImageFilter.GaussianBlur(5))
gaussBlur = img.filter(ImageFilter.Kernel((3, 3), (1/16, 2/16, 1/16, 2/16, 4/16, 2/16, 1/16, 2/16, 1/16), 1, 0))
Sharpen = gaussBlur.filter(ImageFilter.Kernel((3, 3), (0, -1, 0, -1, 5, -1, 0, -1, 0), 1, 0))

filter1 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
apply_filter(img, 1, filter1)
#img.show()
#sobelVer.show()
#sobelHor.show()
#gaussBlur.show()
#Sharpen.show()