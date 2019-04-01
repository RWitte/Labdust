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
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Activation, Dropout
from keras import regularizers
from sklearn.model_selection import train_test_split
from keras import backend as K
import tensorflow as tf

K.clear_session()


### Clear former model
#K.clear_session()
### Calls the sliders function to read folder of .TIF images, make subslices of y, x px length,
### crops to x px (0 for no cropping)
### and generates a 3D array of Image#, x/y values normalized to [0,1]

folder = r"C:\Users\RobertWi\Desktop\151130-AY-artifacts-10x-dapi-gfp-tritc-cy5_Plate_1934\TimePoint_1"
crop_to = 1000
x_len = 100
y_len = 100

dataset = sliders(folder, crop_to, x_len, y_len,2)

#Add channel dimensions, 3D version
def datasetRGB(x):
    x_empty = np.empty((x.shape[0], x.shape[1], x.shape[2], 3), dtype=np.uint8)
    x_empty[:, :, :, 2] = x_empty[:, :, :, 1] = x_empty[:, :, :, 0] = x
    return(x_empty)
    
dataset = datasetRGB(dataset)

#1D version
#dataset = np.reshape(dataset, (dataset.shape[0], dataset.shape[1], dataset.shape[2], 1))

#Normalization to [0,1]
dataset = ((dataset-np.min(dataset))/np.ptp(dataset)).astype('float16')
x_train, x_test = train_test_split(dataset, test_size=0.2, random_state=1)

#x_train = x_train.reshape(-1,(len(x_train), np.prod(x_train.shape[1:])),1)
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
#input_shape = x_train.shape[1]
#print((x_train.shape, x_test.shape))

model = Sequential()

#Encoder Layer
model.add(Conv2D(64, (3, 3),data_format='channels_last', padding='same',))
#model.add(MaxPooling2D(pool_size=(2, 2), input_shape=(32,32,1)))
#model.add(Dropout(0.25))
#model.add(Flatten())
#
#model.add(Dense((4*encoding_dim), activation = 'relu'))
#model.add(Dense((2*encoding_dim), activation = 'relu'))
#model.add(Dense((encoding_dim), activation = 'relu'))
#
##Decoder Layer
#
#model.add(Dense((2*encoding_dim), activation='relu'))
#model.add(Dense((4*encoding_dim), activation='relu'))
#model.add(Dense(10000, activation='sigmoid'))
#
#model.summary()
#
#
#input_img = Input(shape=(2500,))
#encoder_layer1 = model.layers[4]
#encoder_layer2 = model.layers[5]
#encoder_layer3 = model.layers[6]
#encoder = Model(input_img, encoder_layer3(encoder_layer2(encoder_layer1(input_img))))
#
#encoder.summary()
#
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                validation_data=(x_test, x_test))


### Single Lay Encoder starts here
""" 

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

### Plot Results

num_images = 10
np.random.seed(42)
random_test_images = np.random.randint(x_test.shape[0], size=num_images)

encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

plt.figure(figsize=(18, 4))

for i, image_idx in enumerate(random_test_images):
    # plot original image
    ax = plt.subplot(3, num_images, i + 1)
    plt.imshow(x_test[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # plot encoded image
    ax = plt.subplot(3, num_images, num_images + i + 1)
    plt.imshow(encoded_imgs[image_idx].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot reconstructed image
    ax = plt.subplot(3, num_images, 2*num_images + i + 1)
    plt.imshow(decoded_imgs[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

"""
### Single Layer Encoder ends here
"""
### Deep Autoencoder starts here

autoencoder = Sequential()

# Encoder Layers
autoencoder.add(Dense(4 * encoding_dim, input_shape=(input_dim,), activation='relu'))
autoencoder.add(Dense(2 * encoding_dim, activation='relu'))
autoencoder.add(Dense(encoding_dim, activation='relu'))

# Decoder Layers
autoencoder.add(Dense(2 * encoding_dim, activation='relu'))
autoencoder.add(Dense(4 * encoding_dim, activation='relu'))
autoencoder.add(Dense(input_dim, activation='sigmoid'))

autoencoder.summary()


input_img = Input(shape=(input_dim,))
encoder_layer1 = autoencoder.layers[0]
encoder_layer2 = autoencoder.layers[1]
encoder_layer3 = autoencoder.layers[2]
encoder = Model(input_img, encoder_layer3(encoder_layer2(encoder_layer1(input_img))))

encoder.summary()


autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=500,
                validation_data=(x_test, x_test))

### Plot DA results

num_images = 10
np.random.seed(42)
random_test_images = np.random.randint(x_test.shape[0], size=num_images)

encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

plt.figure(figsize=(18, 4))

for i, image_idx in enumerate(random_test_images):
    # plot original image
    ax = plt.subplot(3, num_images, i + 1)
#    plt.imshow(x_test[image_idx].reshape(np.array(dataset_norm).shape[1], np.array(dataset_norm).shape[2]))
    plt.imshow(x_test[image_idx].reshape(100, 100))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # plot encoded image
    ax = plt.subplot(3, num_images, num_images + i + 1)
    plt.imshow(encoded_imgs[image_idx].reshape(int((encoding_dim**0.5)), int((encoding_dim**0.5))))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot reconstructed image
    ax = plt.subplot(3, num_images, 2*num_images + i + 1)
    plt.imshow(decoded_imgs[image_idx].reshape(np.array(dataset_norm).shape[1], np.array(dataset_norm).shape[2]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

### Deep Autoencoder ends here
"""