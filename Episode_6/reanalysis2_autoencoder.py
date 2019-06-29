# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 17:46:10 2019

@author: kylea
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.layers import Dense, Input,GaussianNoise
from keras.layers import Conv3D, MaxPooling3D,UpSampling3D
from keras.models import Model
from keras import backend as K
from reanalysis2_generator import DataGenerator
from keras.callbacks import TensorBoard
import os


K.set_image_data_format('channels_first')
# Parameters years_per_epoch=1,batch_size=32,normalize=True)

padded=True
params = {'years_per_epoch': 1,
          'padded':padded,
          'batch_size': 10,
          'normalization_mode':'min_max',
          'normalize': True}

# Datasets
#partition = # IDs
#labels = # Labels

# Generators
training_generator = DataGenerator(**params)


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
noisy_inputs = GaussianNoise(0.1)(inputs)
x = Conv3D(16, (3, 3, 3), activation='relu', padding='same',data_format='channels_first')(noisy_inputs)
x = MaxPooling3D((2, 2, 2), padding='same',data_format='channels_first')(x)
x = Conv3D(16, (3, 3,3), activation='relu', padding='same',data_format='channels_first')(x)
encoded = MaxPooling3D((2, 2, 2), padding='same', name='encoded',data_format='channels_first')(x)

# Instantiate Encoder Model
encoder = Model(inputs, encoded, name='encoder')
encoder.summary()

decoder1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same',data_format='channels_first',name='decoder1')(encoded)
x2 = UpSampling3D((2, 2, 2),data_format='channels_first')(decoder1)
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


#per this: https://stackoverflow.com/questions/44472693/how-to-decode-encoded-data-from-deep-autoencoder-in-keras-unclarity-in-tutorial
encoded_input = Input(shape=(16,5,20,36))


deco = autoencoder.layers[-5](encoded_input)
deco = autoencoder.layers[-4](deco)
deco = autoencoder.layers[-3](deco)
deco = autoencoder.layers[-2](deco)
deco = autoencoder.layers[-1](deco)

# create the decoder model
decoder = Model(encoded_input, deco)


autoencoder.summary()

autoencoder.compile(loss='mse', optimizer='adam')
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# Train model on dataset
autoencoder.fit_generator(generator=training_generator,
#                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    workers=1,
                    verbose=1,
                    epochs=500,
                    callbacks=[TensorBoard(log_dir='c:\\autoencoder')])

# steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None,
# validation_freq=1, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)


data_dir="T:\\Reanalysis2\\autoencoder_models"
autoencoder_path=os.path.join(data_dir,"autoencoder")
decoder_path=os.path.join(data_dir,"decoder")
encoder_path=os.path.join(data_dir,"encoder")  
autoencoder.save(autoencoder_path)
decoder.save(decoder_path)
encoder.save(encoder_path)




