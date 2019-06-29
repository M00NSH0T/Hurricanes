# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:17:11 2019

inspired by: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

the dimensions of the saved pickles are (variables=6,time~=1464,levels=17,lats=73,lons=144)

@author: kylea
"""

import numpy as np
import pandas as pd
import keras
import os
import pickle
import math
import time

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, years_per_epoch=1,batch_size=32,normalize=True,normalization_mode='min_max',padded=True):
        'Initialization'
        self.batch_size = batch_size
        self.years_per_epoch=years_per_epoch
        self.normalize = normalize
        self.normalization_mode=normalization_mode
        self.padded=padded
        years=range(1979,2013)#leave 2013-2017 as test set
        self.data_dir="T:\Reanalysis2\pkl_files"
        self.year_tracker=pd.DataFrame()
        self.year_tracker['year']=years
        self.year_tracker.index=years
        self.year_tracker['times_used']=0
        self.year_tracker['weights']=1
           
        
        
        
        self.__load_normalize_metrics__()
        self.on_epoch_end()
#        self.__reset__()
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        total_epoch_samples=self.data[1,:,1,1,1].size
        return int(np.floor(total_epoch_samples / self.batch_size))
    
    def __load_year__(self,year):
        file=f"reanalysis2_{year}.pkl"
        full_path=os.path.join(self.data_dir,file)  
        with open(full_path, 'rb') as pickle_file:
            fulldataset=pickle.load(pickle_file)  
        if len(self.data)<=1:#==[] deprecated
            self.data=fulldataset.filled(0)
        else:
            self.data=np.concatenate((self.data,fulldataset.filled(0)),axis=1)
    
    def __select_next_year__(self):
        year=self.year_tracker['year'].sample(n=1,weights=self.year_tracker['weights']).values[0]
        self.year_tracker.loc[year,'times_used']+=1
        if self.year_tracker['times_used'].max()>self.year_tracker['times_used'].min():
            self.year_tracker['weights']=self.year_tracker['times_used'].max()-self.year_tracker['times_used']
        else:
            self.year_tracker['weights']=1
        return year
        
    def __reset__(self):
        self.data=[]
        for y in range(self.years_per_epoch):
            self.__load_year__(self.__select_next_year__())
        
        indices=range(self.data[1,:,1,1,1].size)
        self.sample_tracker=pd.DataFrame()
        self.sample_tracker['indices']=indices
        self.sample_tracker.index=indices
        self.sample_tracker['times_used']=0
        self.sample_tracker['weights']=1 

#    this is the output function that Keras looks for
    def __getitem__(self, indices):
#    def __iter__(self):
        'Generate one batch of data'
        # Generate indexes of the batch


        indices=self.sample_tracker['indices'].sample(n=self.batch_size,weights=self.sample_tracker['weights']).values
        self.sample_tracker.loc[indices,'times_used']+=1
        if self.sample_tracker['times_used'].max()>self.sample_tracker['times_used'].min():
            self.sample_tracker['weights']=self.sample_tracker['times_used'].max()-self.sample_tracker['times_used']
        else:
            self.sample_tracker['weights']=1

        samples=np.swapaxes(np.take(self.data,indices,axis=1),0,1)

        return (samples,samples) #not ideal to send copies of data, but I think this might be the only way to do it (input should equal output for an autoencoder, and a datagenerater expects and X and a y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.__reset__()
        if self.normalize:
            self.__normalizer__()
        if self.padded:
            self.__expand_dims()

#   there should be an easier / more efficient way to do this with np.pad, but I need to figure it out.
    def __expand_dims(self):
        temp=np.zeros((6,self.data.shape[1],20,80,144))
        temp[:,:,:self.data.shape[2],:self.data.shape[3],:]=self.data
        self.data=temp
        
    #placeholder for centering approach
    def __get_storm_coords__(self):
        break
    
#    we do this once, following same method as outlined here: 
#        https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
#    but for a 6 separate 3d grids (one for each variable, which will be analagous to a RGB channel)
#        also see here: https://stackoverflow.com/questions/42460217/how-to-normalize-a-4d-numpy-array
    def __normalizer__(self,layer_size=10512):
        time1=time.time()


        for v in range(6):
            for l in range(17):
                if self.normalization_mode=='min_max':
                    min_val = self.normal_metrics[v,l,2]
                    max_val = self.normal_metrics[v,l,3]
                    nrange=max_val-min_val
                    self.data[v,:,l,:,:] = np.clip(((self.data[v,:,l,:,:] - min_val) / nrange),0,1)             
                else:
                    mean = self.normal_metrics[v,l,0]
                    stddev = self.normal_metrics[v,l,1]
                    adjusted_stddev = max(stddev, 1.0/math.sqrt(layer_size))
                    self.data[v,:,l,:,:] = (self.data[v,:,l,:,:] - mean) / adjusted_stddev
#        self.data=self.data.filled(0)#convert from masked array to np array with 0 replacing missing values (shouldn't be any, but just in case...)
        print("normalizing took: ",time.time()-time1)

        
    def __load_normalize_metrics__(self):
        try:
            file="normalize.pkl"
            full_path=os.path.join(self.data_dir,file)  
            with open(full_path, 'rb') as pickle_file:
                self.normal_metrics=pickle.load(pickle_file)  #6 rows, 2 columns. each row is a variable, col1L mean, col2 stddev
        except:
            print("unable to load normalizer. Calibrating. Please be patient.")
            self.__calibrate_normalization___()
    
    
#    this will calculate the mean, standard deviation, min and max for each variable at each pressure layer and save it to the "normalize.pkl" file
    def __calibrate_normalization___(self):
        self.data=[]
        self.normal_metrics=np.zeros((6,17,4))
        for y in range(5):
            self.__load_year__(self.__select_next_year__())
        for v in range(6):
            for l in range(17):
                self.normal_metrics[v,l,0]=self.data[v,:,l,:,:].mean()
                self.normal_metrics[v,l,1]=self.data[v,:,l,:,:].std()
                self.normal_metrics[v,l,2]=self.data[v,:,l,:,:].min()
                self.normal_metrics[v,l,3]=self.data[v,:,l,:,:].max()                
            
        file="normalize.pkl"
        full_path=os.path.join(self.data_dir,file)  
        with open(full_path, 'wb') as pickle_file:
            pickle.dump(self.normal_metrics,pickle_file,protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
if __name__=='__main__':
        #inserted due to https://stackoverflow.com/questions/45720153/python-multiprocessing-error-attributeerror-module-main-has-no-attribute
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    d=DataGenerator()
#    d.__calibrate_normalization___()