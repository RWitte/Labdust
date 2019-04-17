"""
Autoencoder setup based on: https://ramhiser.com/post/2018-05-14-autoencoders-with-keras/
"""
from __future__ import division
import numpy as np
import cv2
import glob as glob
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Activation, Dropout
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras import backend as K
from builtins import map
import sys

#from matplotlib import pyplot as plt
#from PIL import Image, SVG

K.clear_session()

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
    
# slice images, blockshape in y/x dimensionality, generation of 5D array 
#(image #, row #, column #, x coordinates, y coordinates)

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

def blockwise_view( a, blockshape, aslist=False):

    blockshape = tuple(blockshape)
    outershape = tuple(np.array(a.shape) // blockshape)
    view_shape = outershape + blockshape

    # inner strides: strides within each block (same as original array)
    intra_block_strides = a.strides

    # outer strides: strides from one block to another
    inter_block_strides = tuple(a.strides * np.array(blockshape))

    # This is where the magic happens.
    # Generate a view with our new strides (outer+inner).
    view = np.lib.stride_tricks.as_strided(a,
                                              shape=view_shape, 
                                              strides=(inter_block_strides+intra_block_strides))

    if aslist:
        return list(map(view.__getitem__, np.ndindex(outershape)))
    return view

#Add channel dimension
def add_channels(x):
    x_empty = np.empty((x.shape[0], x.shape[1], x.shape[2], channels), dtype=np.uint8)
    i = 0
    while i < channels:
        x_empty[:,:,:,i] = x
        i += 1
    return(x_empty)

### Calls the sliders function to read folder of .TIF images, make subslices of y, x px length,
### crops to x px (0 for no cropping)
### and generates a 3D array of Image#, x/y values normalized to [0,1]

folder = r"C:\Users\Robert\Documents\GitHub\Labdust\TimePoint_1"
crop_to = 1000
x_len = 100
y_len = 100
image_num = 2 #(leave empty for all)
channels = 1

dataset = sliders(folder, crop_to, x_len, y_len, image_num)
dataset = add_channels(dataset)

#Normalization to [0,1], split into train/testset
dataset = ((dataset-np.min(dataset))/np.ptp(dataset)).astype('float32')
x_train, x_test = train_test_split(dataset, test_size=0.2, random_state=1)


#Autoencoder as function

batch_size = 50
epochs = 50

input_img = Input(shape = (x_len, y_len, channels))
input_shape = (x_len, y_len, channels)

def autoencoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)

    #decoder
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded

autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
autoencoder.summary()
#sys.exit()

autoencoder_train = autoencoder.fit(x_train, x_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, x_test))

#model = Sequential()
##Encoder Layer
#model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="valid",input_shape=input_shape))
#model.add(Conv2D(32, (32,32),data_format='channels_last', activation ="relu", padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2),))
#model.add(Conv2D(64, (3, 3),data_format='channels_last', activation ="relu", padding='same',))
#model.add(MaxPooling2D(pool_size=(2, 2),))
###Decoder Layer
#model.add(Conv2D(32, (3, 3),data_format='channels_last', activation ="relu", padding='same',))
#model.add(UpSampling2D((2,2)))
#model.add(Conv2D(1, (3,3), activation = 'sigmoid', padding = 'same'))
#
#model.compile(loss='mean_squared_error', optimizer='RMSprop')
#
#model.fit(x_train, x_train,batch_size=batch_size, epochs=epochs, validation_data=(x_test, x_test))