# -*- coding: utf-8 -*-

import numpy as np
import cv2
import glob as glob
from blockwise_view2 import blockwise_view
from matplotlib import pyplot as plt
from PIL import Image

def sliders(folder, crop_to = 1000, x_len = 100, y_len = 100, num = None):
### Input folder, "r" for raw string, , load in .TIF files, replace "\" with "/"
    folder = folder.replace("\\", "/")
    folder += r"/**/*.TIF"

### Read images from folder

    images = []
    files = glob.glob(folder,recursive=True)
    for myFile in files[:num]:
        image = cv2.imread(myFile, -1).astype('uint8')
    #   image = image.astype(float)
    # Crop to innermost 1000x1000 px
        if crop_to:
            if crop_to > min(np.shape(image)[0],np.shape(image)[1]):
                crop_to = min(np.shape(image)[0],np.shape(image)[1])
            image = image[int(((np.shape(image)[0])/2)-int(crop_to/2)):int(((np.shape(image)[0])/2)+int(crop_to/2)),
                          int(((np.shape(image)[1])/2)-int(crop_to/2)):int(((np.shape(image)[1])/2)+int(crop_to/2))]
             
        images.append(image.astype("uint8"))
    
#print('Array shape:', np.array(images).shape)

### slice images, blockshape in y/x dimensionality, generation of 5D array (image #, row #, column #, x coordinates, y coordinates)

    slices = []
    for i in range(0, len(images)):
        slice = blockwise_view(images[i], blockshape = (x_len,y_len))
        slices.append(slice)

    image_array = np.array(slices).astype('uint8')
    dataset = image_array.reshape(-1,x_len,y_len)
    return(dataset)
    
"""
### Save slices to file, i image number, j row number, k column number
    image_num = np.array(image_array).shape[0]
    rows = np.array(image_array).shape[1]
    columns = np.array(image_array).shape[2]


for i in range (0,image_num):
    for j in range(0,rows):
        for k in range(0,columns):
            slice = (str(i+1) + " " + str(j+1) + "_" + str(k+1))
            print(str(i+1) + " " + str(j+1) + " " + str(k+1))
            plt.imshow(image_array[i, j, k], interpolation='nearest')
            plt.show()
            img = Image.fromarray(image_array[i, j, k])
            img.save(str(i+1) + "_" + str(j+1) + "_" + str(k+1) + ".TIF")
"""