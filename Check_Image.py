# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:47:54 2019

@author: Robert
"""

from PIL import Image
import numpy as np

w, h = 100, 100
data = dataset[0,:,:,0].astype("uint8")
img = Image.fromarray(data)
img.save('my.png')
img.show()