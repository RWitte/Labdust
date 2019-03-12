# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2
import glob as glob

### Input folder, "r" for raw string, , load in .TIF files, replace "\" with "/"
folder = r"C:\Users\Robert\Desktop\labdust-gan\151130-AY-artifacts-4x-dapi-gfp-tritc-cy5_Plate_1935\TimePoint_1"
folder += r"\*.TIF"
folder = folder.replace("\\", "/")

images = []
files = glob.iglob(folder)
for myFile in files:
    image = cv2.imread(myFile, -1)
    images.append(image)

#print('Array shape:', np.array(images).shape)

#Convert list of arrays into 3D np.array

###x = np.zeros((np.array(images).shape[0],np.array(images).shape[1],np.array(images).shape[2]))

images = np.array(images)

"""
###Look at images
from matplotlib import pyplot as plt
plt.imshow(images[0], interpolation='nearest')
plt.show()
"""

