# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 18:01:14 2019

This is the learner for the cluster approach I discuss in Episode 4. To make this work, you
need both a populated MySQL database (or an alternative SQL db... I recommend not using SQLite 
because you can't write to it from multiple threads at once). 

@author: kylea
"""

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import LeakyReLU,Concatenate
from keras import backend as K
import math
from keras.optimizers import Adagrad,Adam

import numpy as np
from sqlalchemy import create_engine
import pandas as pd
import pika
import pickle
import time
import requests
import json
import tensorflow as tf
import os    
import time
from keras.models import load_model

from keras.backend.tensorflow_backend import set_session

#https://github.com/keras-team/keras/issues/1538
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.65
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0" #only the gpu 0 is allowed
#config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))



conn = create_engine('mysql://your_username:your_password@localhost:3306/weather')#if running a cluster setup, replaced localhost with local ip address
K.set_image_data_format('channels_first')

train_test_split=pd.read_csv('train_test_split.csv')
train=train_test_split.loc[train_test_split.set=='train','Hurricane_ID']
total_hurricanes=len(train)

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'#this prevents the client using the gpu. only use if running on same machine as server.


class learner:
    """this object will continuously upload new samples to a rabbitmq queue, to be pulled by the learner for training.
    """
    def __init__(self, storm_list=pd.Series(['AL032006','AL082006','AL062008','AL112008']),timesteps_per_sample=4,batch_size=15,epochs=20000):
        """
        hurricane_list= list of hurricanes to iterate through and send to queue. May want to do a few of variations of this sample_generator - one for each train, test and validation set
        """
        self.start_pika()
        self.create_model()
        self.storm_list=storm_list
        self.timesteps_per_sample=timesteps_per_sample
        self.batch_size=batch_size
        self.hurricanes_per_pass=30
        self.epochs=epochs
        self.passes_per_epoch=int(total_hurricanes/self.hurricanes_per_pass)
        
    def start_pika(self):
        self.credentials=pika.PlainCredentials('moonshot','episode3')
        self.parameters=pika.ConnectionParameters(host='localhost',port=5672,credentials=self.credentials)
        self.connection = pika.BlockingConnection(parameters=self.parameters)
        self.channel = self.connection.channel()
        self.routing_key='weather'
        self.channel.exchange_declare(exchange=self.routing_key,
                         exchange_type='direct')
        self.channel.queue_declare(queue=self.routing_key)      
        self.channel.queue_bind(exchange=self.routing_key,
                   queue=self.routing_key)     

    def create_model(self):
        """
        need to create a network that can essentially handle three channel input, similar to RGB, except
        channel 1 = u wind component, channel 2 = v wind component, and channel 3 = all zero matrix with single active
        pixel at current center of storm, or perhaps a circle centered on that point with the radius determined by other
        factors in the hurricane path table
        """
        pass
        
        input1 = Input(shape=(3,309,720))  
        layer1=Conv2D(32, (3, 3), padding='same')(input1)
        layer1=LeakyReLU()(layer1)
        layer2=MaxPooling2D((2, 2), strides=(2, 2))(layer1)
        layer3=Conv2D(64, (3, 3), padding='same')(layer2)
        layer3=LeakyReLU()(layer3)
        layer4=MaxPooling2D((2, 2), strides=(2, 2))(layer3)  
        
#        layer3=Convolution3D(256, (3, 3, 3), padding='same')(layer2)
#        layer3=leaky_relu()(layer3)
#        layer4=MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(layer3)
        layer5=Flatten()(layer4)
        

        input2 = Input(shape=(2,))  
        merged_layer=Concatenate()([layer5,input2])
        layer6=Dense(32,activation='linear')(merged_layer)
        layer6=LeakyReLU()(layer6)
        layer7=Dense(16,activation='linear')(layer6)
        layer7=LeakyReLU()(layer7)
        layer8=Dense(8,activation='linear')(layer7)
        layer8=LeakyReLU()(layer8)        
        layer9=Dense(4,activation='linear')(layer8)
        layer9=LeakyReLU()(layer9)          
        output=Dense(2,activation='linear')(layer9)
        self.model = Model(inputs=[input1,input2], outputs=output)
#        adagrad = Adagrad()#lr=0.0002)
        adam=Adam(clipnorm=0.5)
#        run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
#        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        self.model.compile(loss='mean_squared_error', optimizer = adam, metrics=['mse'])#,options = gpu_options)
        print(self.model.metrics_names)
        try:
            self.load_model()
            print("successfully loaded model")
        except:
            print("no model weights found.")
        
    def next_pass(self):
        """
        this pulls a given number of hurricanes from the queue and combines them into a memory array (or set of U,V and y arrays), which
        will be sampled to get batches.
        
        There are a few approaches that could be taken here to form a batch. One would be to form a replay memory or 
        priority replay memory as is commonly done with reinforcement learning, where samples are read of the queue and deposited 
        into memory for more rapid sampling. With a priority replay memory, samples that result in larger error during train_on_batch calls
        would be given a higher likelihood of being sampled again later. Once memory limit is reached, new samples pulled from queue overwrite the
        oldest. This approach is worth pursuing, but for the sake of expediency in developing a rapid prototype, we're going to do something
        simpler here. Instead of creating a memory object, we're simply going to ensure that we pull a few difference hurricanes into the each batch.
        This should help the neural network remain stable and not keep being pulled toward overfitting toward the global weather patterns surrounding 
        each hurricane if all samples were from a single hurricane at a time. 
        
        Another, potentially mid-path solution would be to create a new table, perhaps just a pandas table that gets generated in the initilization of this
        class, which contains the error associated with each timestep/hurricane pair. 
        """
        
        Us=[]
        Vs=[]
        yps=[]
        ys=[]
#        while self.get_queue_length()<(self.hurricanes_per_pass*2):
#            time.sleep(0.1)
        for i in range(0,self.hurricanes_per_pass):
            U,V,yp,y=self.get_next_hurricane()
            Us=Us+U
            Vs=Vs+V
            yps=yps+yp
            ys=ys+y
        return Us,Vs,yps,ys
        
    
#    def get_single_sample(self):
#        """
#        timesteps_per_sample indicates number of past timesteps to be used to make a prediction. For instance,
#        if timesteps_per_sample=1, then we just train the network to look at the current wind state and extrapolate from there.
#        if timesteps_per_sample=2, then we take the current state, plus the last state (6 hours prior), etc.
#        """
#        pass
    
    def get_next_hurricane(self,queue='weather'):
        """
        grab the next hurricane
        """
        while True:
            try:
                method_frame, header_frame, body = self.channel.basic_get(queue)
                Us,Vs,yps,ys=pickle.loads(body)
                self.channel.basic_ack(method_frame.delivery_tag)
                break
            except:
                print('Error getting next timestep. Restarting pika.')
                self.start_pika()
#                method_frame, header_frame, body = self.channel.basic_get(queue)
#                Us,Vs,yps,ys=pickle.loads(body)
#                self.channel.basic_ack(method_frame.delivery_tag)
        return Us,Vs,yps,ys
        
    def get_index_from_coord(self,lat,lon):
        lat_r=round(lat*2)/2 #round to nearest half-degree
        lon_r=round(lon*2)/2#round to nearest half-degree
        x=int(0+lon_r*2)  #when lon=0, index=0. when lon=359.5, index=719 (360,720) 
        y=int(154-lat_r*2) #when lat=77, index=0. when lat=-77, index=308
        return x,y
    
    def run(self):

        #this structure just lets the generator run indefinitely, until keyboard interrupt called. In the future,
        #we might want to do something where we trigger it to end by sending a signal from the learner informing
        #the generator threads that it's completed training. But if we're careful to cap the max queue lengths,
        #manually closing them when done is a minor inconvenience at most. 
        try:
            step=max(pd.read_sql('training_loss_adam_clipped',conn).step)
        except:
            step=0

        try:
#            while True:
            for i in range(0,self.epochs):
                for j in range(0,self.passes_per_epoch):
                    while self.get_queue_length()<(self.hurricanes_per_pass+5):  
                        print("queue only ",self.get_queue_length()," / ",self.hurricanes_per_pass+5," ... waiting")
                        time.sleep(10)
                    Us,Vs,yps,ys=self.next_pass()
                    passes=0
                    time.sleep(5)
                    while self.get_queue_length()<(self.hurricanes_per_pass+5):      
                        X,yps_sample,y=self.get_sample(Us,Vs,yps,ys)
                        loss,metrics=self.model.train_on_batch([X,yps_sample],y)
                        print("epoch: ", i,":  loss: ",loss)
                        step+=1
                        loss_df=pd.DataFrame([[step,loss]],columns=['step','loss'])
#                        loss_df=pd.DataFrame([[step,loss]],columns=['loss','mean_square_error'])

                        loss_df.to_sql('training_loss_adam_clipped',conn,index=False,if_exists='append')
                        
                        passes+=1
                    print("number of passes: ",passes)
                    step+=1
                    3
                self.save_model()
        except KeyboardInterrupt:
            print('exiting loop')
            self.save_model()
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that


    def get_sample(self,Us,Vs,yps,ys):
        X=[]
        y=[]
        yps_out=[]
        for i in np.random.choice(len(ys), self.batch_size):
            X_ch3=np.zeros([309,720],dtype=np.int16)
            index1,index2=self.get_index_from_coord(yps[i].Lat.values[0],yps[i].Lon.values[0])
            X_ch3[index2,index1]=255
            X.append(np.stack([Us[i].values,Vs[i].values,X_ch3]))
            y.append(np.array([ys[i].Lat.values,ys[i].Lon.values]))
            yps_out.append([yps[i].Lat.values[0],yps[i].Lon.values[0]])

        return np.stack(X,axis=0),np.stack(yps_out,axis=0),np.squeeze(np.stack(y,axis=0)),
    
    def load_model(self):
        try:
            self.model= load_model("model.h5")
        except:
            self.model.load_weights("weights.hd5")
    
    def save_model(self):
        self.model.save("model.h5")
        self.model.save_weights("weights.hd5")
        print("model saved")

    def get_queue_length(self,queue='weather'):
        user = 'moonshot'
        passwd = 'episode3'   
        host = 'localhost'
        port = 15672
        url = 'http://%s:%s/api/queues' % (host, port)
        res = requests.get(url, auth=(user,passwd))
        data = json.loads(res.content.decode('unicode_escape'))
        length=0
        for d in data:
            if d['name']==queue:
                length=d['messages']
        return length




if __name__ == '__main__':
    
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    learner=learner()
    

 
    learner.run()
  


