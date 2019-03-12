# -*- coding: utf-8 -*-
"""
Sliding function for subsampling images
"""

import numpy as np
import cv2
import os

### Input folder, replace "\" with "/", "r" for raw string
folder = r"C:\Users\Robert\Desktop\labdust-gan\151130-AY-artifacts-4x-dapi-gfp-tritc-cy5_Plate_1935\TimePoint_1"

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename,),-1)
        if img is not None:
            images.append(img)
    return images

Images = load_images_from_folder(folder)