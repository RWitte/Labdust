"""
Autoencoder setup based on: https://ramhiser.com/post/2018-05-14-autoencoders-with-keras/
"""
from sliders_fun import sliders

from IPython.display import Image, SVG
import matplotlib.pyplot as plt

import numpy as np
from keras.models import Model, Sequential
from keras.optimizers import RMSprop
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Activation, Dropout
from sklearn.model_selection import train_test_split
from keras import backend as K

K.clear_session()


### Clear former model
#K.clear_session()
### Calls the sliders function to read folder of .TIF images, make subslices of y, x px length,
### crops to x px (0 for no cropping)
### and generates a 3D array of Image#, x/y values normalized to [0,1]

folder = r"C:\Users\Robert\Documents\GitHub\Labdust\TimePoint_1"
crop_to = 1000
x_len = 101
y_len = 101

dataset = sliders(folder, crop_to, x_len, y_len,2)

#Add channel dimensions, 3D version
#def datasetRGB(x):
#    x_empty = np.empty((x.shape[0], x.shape[1], x.shape[2], 3), dtype=np.uint8)
#    x_empty[:, :, :, 2] = x_empty[:, :, :, 1] = x_empty[:, :, :, 0] = x
#    return(x_empty)
#    
#dataset = datasetRGB(dataset)

#1D version
dataset = np.reshape(dataset, (dataset.shape[0], dataset.shape[1], dataset.shape[2], 1))

#Normalization to [0,1]
dataset = ((dataset-np.min(dataset))/np.ptp(dataset)).astype('float32')
x_train, x_test = train_test_split(dataset, test_size=0.2, random_state=1)

batch_size = 32
epochs = 50
inChannel = 1
x, y = 100, 100
input_img = Input(shape = (x, y, inChannel))

#def autoencoder(input_img):
#    #encoder
#    #input = 28 x 28 x 1 (wide and thin)
#    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
#    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
#    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
#    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
#    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
#
#    #decoder
#    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
#    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
#    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
#    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
#    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
#    return decoded
#
#autoencoder = Model(input_img, autoencoder(input_img))
#autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
#autoencoder.summary()
#
#autoencoder_train = autoencoder.fit(x_train, x_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, x_test))

#
model = Sequential()
###Encoder Layer
model.add(Conv2D(1, (32,32),data_format='channels_last', activation ="relu", padding='same', input_shape=(101,101,1)))
model.add(Conv2D(32, (32,32),data_format='channels_last', activation ="relu", padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2),))
#model.add(Conv2D(64, (3, 3),data_format='channels_last', activation ="relu", padding='same',))
#model.add(MaxPooling2D(pool_size=(2, 2),))
####Decoder Layer
#model.add(Conv2D(32, (3, 3),data_format='channels_last', activation ="relu", padding='same',))
#model.add(UpSampling2D((2,2)))
#model.add(Conv2D(1, (3,3), activation = 'sigmoid', padding = 'same'))

model.compile(optimizer='RMSprop', loss='mean_squared_error')

model.fit(x_train, x_train,
                epochs=50,
                batch_size=32,
                validation_data=(x_test, x_test))