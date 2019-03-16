"""
Autoencoder setup based on: https://ramhiser.com/post/2018-05-14-autoencoders-with-keras/
"""

from sliders_fun import sliders

from IPython.display import Image, SVG
import matplotlib.pyplot as plt

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras import regularizers
from sklearn.model_selection import train_test_split

### Calls the sliders function to read folder of .TIF images, make subslices of y/x dimension 
### and generates a 3D array of Image#, x/y values normalized to [0,1]

folder = r"C:\Users\RobertWi\Desktop\151130-AY-artifacts-10x-dapi-gfp-tritc-cy5_Plate_1934\TimePoint_1"
dataset_norm = sliders(folder, 100,100)

x_train, x_test = train_test_split(dataset_norm, test_size=0.2, random_state=1)

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
input_shape = x_train.shape[1]
print((x_train.shape, x_test.shape))

input_dim = x_train.shape[1]
encoding_dim = 400

compression_factor = float(input_dim) / encoding_dim
print("Compression factor: %s" % compression_factor)

autoencoder = Sequential()
autoencoder.add(
    Dense(encoding_dim, input_shape=(input_dim,), activation='relu')
)
autoencoder.add(
    Dense(input_dim, activation='sigmoid')
)

autoencoder.summary()

input_img = Input(shape=(input_dim,))
encoder_layer = autoencoder.layers[0]
encoder = Model(input_img, encoder_layer(input_img))

encoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=100,
                shuffle=True,
                validation_data=(x_test, x_test))