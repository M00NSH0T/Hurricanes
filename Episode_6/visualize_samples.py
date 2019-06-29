# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:32:28 2019

@author: kylea
"""
import pickle
from keras.models import load_model
import os
import numpy as np
import cv2

#change to your own personal folder location
data_dir="T:\Reanalysis2\pkl_files"

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'#this prevents the client using the gpu. only use if running on same machine as server.


#opens the pickle that contains the converted numpy arrays of spatial data
def load_year(year):
    file=f"reanalysis2_{year}.pkl"

    full_path=os.path.join(data_dir,file)  
    with open(full_path, 'rb') as pickle_file:
        fulldataset=pickle.load(pickle_file)  
    data=fulldataset.filled(0)
    return data

#pads the data in the same way that the generator did. dimensions must be divisible by 4 (divisible by 2, twice), for the decoder output dimensions to match encoder input dimensions.
#    we could have cropped it instead
def reformat(data):
    temp=np.zeros((6,data.shape[1],20,80,144))
    temp[:,:,:data.shape[2],:data.shape[3],:]=data
    data=np.swapaxes(temp,0,1)
#    data = np.reshape(data, (data.shape[0], -1))
    return data

#this will load the normalization metrics used by the generator during training to scale input data
#    in a similar way to MinMaxScaler in sklearn
def normalize(data):
    file="normalize.pkl"
    full_path=os.path.join(data_dir,file)  
    with open(full_path, 'rb') as pickle_file:
        normal_metrics=pickle.load(pickle_file)  #6 rows, 2 columns. each row is a variable, col1L mean, col2 stddev
    for v in range(6):
        for l in range(17):
            min_val = normal_metrics[v,l,2]
            max_val = normal_metrics[v,l,3]
            nrange=max_val-min_val
            data[v,:,l,:,:] = np.clip(((data[v,:,l,:,:] - min_val) / nrange),0,1)
    return data


autoencoder=load_model("T:\\Reanalysis2\\autoencoder_models\\autoencoder")


y=2013
print(y)
data=load_year(y)
data=normalize(data)
In=reformat(data)
Out=autoencoder.predict(In)   

variables=['air','hgt','omega','rhum','uwnd','vwnd']

#the following will generate two images... a selected channel (variable) for a given pressure layer (X), and then the corresponding 
#encoded / decoded version for comparison (Out)


for i,var in enumerate(variables):
    input_sample=255*In[i,3,1,:,:]
    cv2.imwrite(f"input_{var}.jpg",input_sample)
    
    
    output_sample=255*Out[i,3,1,:,:]
    cv2.imwrite(f"output_{var}.jpg",output_sample)