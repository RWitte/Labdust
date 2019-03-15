# -*- coding: utf-8 -*-

import numpy as np
import cv2
import glob as glob
from blockwise_view import blockwise_view
from matplotlib import pyplot as plt
from PIL import Image

def sliders(folder, x_len = 100, y_len = 200):
### Input folder, "r" for raw string, , load in .TIF files, replace "\" with "/"
    folder = folder.replace("\\", "/")
    folder += r"/*.TIF"

### Read images.

    images = []
    files = glob.iglob(folder)
    for myFile in files:
        image = cv2.imread(myFile, -1)
        image = image.astype(float)
        images.append(image)

#print('Array shape:', np.array(images).shape)

### slice images, blockshape in y/x dimensionality, generation of 5D array (image #, row #, column #, x coordinates, y coordinates)

    slices = []
    for i in range(0, len(images)):
        slice = blockwise_view(images[i], blockshape = (x_len,y_len), require_aligned_blocks = False)
        slices.append(slice)

    image_array = np.array(slices)

    image_num = np.array(image_array).shape[0]
    rows = np.array(image_array).shape[1]
    columns = np.array(image_array).shape[2]
    dataset = image_array.reshape(-1,x_len,y_len)
    dataset_norm = (dataset-np.min(dataset))/np.ptp(dataset)
    return(dataset_norm)

"""
### Save slices to file, i image number, j row number, k column number
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

