# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 17:46:10 2019

@author: kylea
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.layers import Dense, Input,GaussianNoise
from keras.layers import Conv3D, Flatten,MaxPooling3D,UpSampling3D,ZeroPadding3D
from keras.layers import Reshape,Dense
from keras.models import Model
from keras import backend as K
import numpy as np
from reanalysis2_generator import DataGenerator
from keras.callbacks import TensorBoard



K.set_image_data_format('channels_first')
# Parameters years_per_epoch=1,batch_size=32,normalize=True)

padded=True
params = {'years_per_epoch': 1,
          'padded':padded,
          'batch_size': 10,
          'normalize': True}

# Datasets
#partition = # IDs
#labels = # Labels

# Generators
training_generator = DataGenerator(**params)

#need to add this training / validation split back in to prevent overfitting, but 
#that's going to require some work. 
#validation_generator = DataGenerator(partition['validation'], labels, **params)

# Design model
#noise = np.random.normal(loc=0.5, scale=0.5, size=(6,17, 93,144))
#x_train_noisy = x_train + noise
#noise = np.random.normal(loc=0.5, scale=0.5, size=(6,17, 93,144))
#x_test_noisy = x_test + noise
#
#x_train_noisy = np.clip(x_train_noisy, 0., 1.)
#x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Network parameters
if padded:
    input_shape = (6,20, 80,144) #padded with zeros so dividing by 2 twice won't result in odd number and strange decoder behavoir
else:
    input_shape = (6,17, 73,144) #6 channels (variables, 17 height layers, 73 latitudes, 144 longitudes)
batch_size = params['batch_size']
kernel_size = 3
latent_dim = 256
# Encoder/Decoder number of CNN layers and filters per layer
layer_filters = [32, 64]

# Build the Autoencoder Model
# First build the Encoder Model
inputs = Input(shape=input_shape, name='inputs')
noisy_inputs = GaussianNoise(0.2)(inputs)
#x= ZeroPadding3D(padding=(1, 1, 0), data_format='channels_first')(noisy_inputs)
#x = noisy_inputs

x = Conv3D(16, (3, 3, 3), activation='relu', padding='same',data_format='channels_first')(noisy_inputs)
x = MaxPooling3D((2, 2, 2), padding='same',data_format='channels_first')(x)
x = Conv3D(16, (3, 3,3), activation='relu', padding='same',data_format='channels_first')(x)
x = MaxPooling3D((2, 2, 2), padding='same', name='encoded',data_format='channels_first')(x)
x=Flatten()(x)
x=Dense(4096, activation='relu')(x)
encoded=Dense(4096, activation='relu')(x)
# Instantiate Encoder Model
encoder = Model(inputs, encoded, name='encoder')
encoder.summary()

#x2=Dense(4096, activation='relu')()
first_decoder_layer=Dense(4096, activation='relu',name='first_decoder_layer')(encoded)
x2=Dense(57600, activation='relu')(first_decoder_layer)
x2=Reshape((16,5,20,36))(x2)
x2 = Conv3D(16, (3, 3, 3), activation='relu', padding='same',data_format='channels_first')(x2)
x2 = UpSampling3D((2, 2, 2),data_format='channels_first')(x2)
x2 = Conv3D(16, (3, 3, 3), activation='relu', padding='same',data_format='channels_first')(x2)
x2 = UpSampling3D((2, 2, 2),data_format='channels_first')(x2)




decoded = Conv3D(6, (3, 3, 3), activation='sigmoid', padding='same',name='decoded',data_format='channels_first')(x2)

#outputs = Activation('sigmoid', name='decoder_output')(x)

# Instantiate Decoder Model
# create a placeholder for an encoded (32-dimensional) input

# Autoencoder = Encoder + Decoder
# Instantiate Autoencoder Model
autoencoder = Model(inputs, decoded, name='autoencoder')
#autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')



#encoded_input = Input(shape=(16,5,20,36))
#encoded_input = Input(shape=(4096,))
### retrieve the last layer of the autoencoder model
#decoder_layer = autoencoder.layers[-1]
### create the decoder model
#decoder = Model(encoded_input, first_decoder_layer(encoded_input))



autoencoder.summary()

autoencoder.compile(loss='mse', optimizer='adam')
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# Train model on dataset
autoencoder.fit_generator(generator=training_generator,
#                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    workers=1,
                    verbose=1,
                    epochs=160,
                    callbacks=[TensorBoard(log_dir='c:\\autoencoder')])

# steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None,
# validation_freq=1, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)






# at this point the representation is (7, 7, 32)



