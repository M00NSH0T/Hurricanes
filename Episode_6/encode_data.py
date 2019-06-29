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
from datetime import datetime,timedelta
import pandas as pd
from sqlalchemy import create_engine

#change to your own personal folder location
data_dir="T:\Reanalysis2\pkl_files"

#make sure to update your credentials (and make something secure)
conn = create_engine('mysql://username:password@localhost:3306/weather')


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


encoder=load_model("T:\\Reanalysis2\\autoencoder_models\\encoder")
autoencoder=load_model("T:\\Reanalysis2\\autoencoder_models\\autoencoder")

train_years=range(1979,2013)#leave 2013-2017 as test set
test_years=range(2013,2018)

#all_data=pd.DataFrame()
all_data=[]

for y in range(1979,2019):
    print(y)
    data=load_year(y)
    data=normalize(data)
    X=reformat(data)
    encoded=encoder.predict(X)
    encoded = np.reshape(encoded, (encoded.shape[0], -1))
    dts=[datetime(y,1,1)+n*timedelta(hours=6) for n in range(len(encoded))]
    temp=pd.DataFrame(encoded,index=dts)
#    temp.to_sql('encoded_reanalysis2_v1',conn,if_exists='append',index=True)
    all_data.append(temp)
    
    
all_data=pd.concat(all_data)

file_out="encoded_data.pkl"
full_path=os.path.join(data_dir,file_out)
all_data.to_pickle(full_path)



#the following will generate two images... a selected channel (variable) for a given pressure layer (X), and then the corresponding 
#encoded / decoded version for comparison (Out)
#edit: use "visualize_samples.py" to get a complete set instead.
#input_sample=255*X[1,3,1,:,:]
#
#cv2.imwrite('input_sample2.jpg',input_sample)
#
#Out=autoencoder.predict(X)
#output_sample=255*Out[1,3,1,:,:]
#cv2.imwrite('output_sample2.jpg',output_sample)