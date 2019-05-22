# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 06:39:22 2018

@author: kylea
"""


import numpy as np
from sqlalchemy import create_engine
import pandas as pd
import pika
import pickle
import time
import requests
import json
from datetime import timedelta
from sklearn.utils import shuffle

conn = create_engine('mysql://your_username:your_password@localhost:3306/weather')#if running a cluster setup, replaced localhost with local ip address
train_test=pd.read_csv('train_test_split.csv')

class sample_generator:
    """this object will continuously upload new samples to a rabbitmq queue, to be pulled by the learner for training.
    """
    def __init__(self, buffer_length=40,storm_list=pd.Series(['AL032006','AL082006','AL062008','AL112008'])):
        """
        buffer_length=how many hurricanes' worth of data to queue up at a time. More will push memory limits. too few will cause things to run slowly.
        hurricane_list= list of hurricanes to iterate through and send to queue. May want to do a few of variations of this sample_generator - one for each train, test and validation set
        """
        self.start_pika()
        self.buffer_length=buffer_length
        self.storm_list=storm_list
        
        
    def start_pika(self):
        self.credentials=pika.PlainCredentials('moonshot','episode3')
        self.parameters=pika.ConnectionParameters(host='localhost',port=25672,credentials=self.credentials)
        self.connection = pika.BlockingConnection(parameters=self.parameters)
        self.channel = self.connection.channel()
        self.routing_key='weather'
        self.channel.exchange_declare(exchange=self.routing_key,
                         exchange_type='direct')
        self.channel.queue_declare(queue=self.routing_key)      
        self.channel.queue_bind(exchange=self.routing_key,
                   queue=self.routing_key)     

  
    def send(self,message):
        pickled_message=pickle.dumps(message) 
        try:
            self.channel.basic_publish(exchange='',
                              routing_key='weather',
                              body=pickled_message)
        except:
            self.connection = pika.BlockingConnection(parameters=self.parameters)
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue='weather')
            self.channel.basic_publish(exchange='',
                              routing_key='weather',
                              body=pickled_message)
        
    def run(self):

        #this structure just lets the generator run indefinitely, until keyboard interrupt called. In the future,
        #we might want to do something where we trigger it to end by sending a signal from the learner informing
        #the generator threads that it's completed training. But if we're careful to cap the max queue lengths,
        #manually closing them when done is a minor inconvenience at most. 
        storms=self.storm_list
        try:
            while True:
                storms=shuffle(storms)
                for storm in storms:
                    print(storm)
                    a=self.get_hurricane_data(hurricane_id=storm)
                    print('sent')
                    try:
                        self.send(a)
                    except:
                        self.start_pika()
                        self.send(a)
                    current_queue=self.get_queue_length()
                    while current_queue>self.buffer_length:
                        print('long queue... waiting to continue.')
                        time.sleep(10)
                        current_queue=self.get_queue_length()
    
        except KeyboardInterrupt:
            print('exiting loop')
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that


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

    def get_hurricane_data(self,hurricane_id='AL192017'):
        query_1="select Date_Time,Adj_Lat as Lat,Adj_Lon as Lon from track_data where Hurricane_ID=%s"
        y=pd.read_sql(query_1,conn,params=[hurricane_id])
        y['last_timestep']=y['Date_Time']-timedelta(hours=6)
#        y=y[['last_timestep','Max_Speed','Adj_Lat','Adj_Lon']]
        datetimes=y.loc[:,'Date_Time'].drop_duplicates()
        
        datetimes=tuple(datetimes.apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S")))
    
        Us=[]
        Vs=[]
        y_p=[]
        y_f=[]
        for dt in datetimes:
            
            query_2 = 'SELECT * FROM grib_data_stacked WHERE datetime=%s'
            X=pd.read_sql(query_2,conn,params=[dt])
            X_temp=X.loc[X.datetime==dt].copy()
            U=X_temp.drop(columns=['datetime','v'],inplace=False).copy()
            U.set_index(['lat','lon'],inplace=True)
            U=U.unstack(level=-1)
            U.sort_index(ascending=False,inplace=True)
            adjustment_factor=40
            U.fillna(-adjustment_factor,inplace=True)
            
    #        X_temp=X.loc[X.datetime==dt]
            V=X_temp.drop(columns=['datetime','u'],inplace=False).copy()
            V.set_index(['lat','lon'],inplace=True)
            V=V.unstack(level=-1)
            V.sort_index(ascending=False,inplace=True)
            V.fillna(-adjustment_factor,inplace=True)
            yp_out=y[y.Date_Time==dt]            
            yf_out=y[y.last_timestep==dt]
            if (len(U)>0 and len(V)>0 and len(yp_out)>0 and len(yf_out)>0):
                Us.append(U)
                Vs.append(V)
                y_p.append(yp_out)
                y_f.append(yf_out)
        
        return Us,Vs,y_p,y_f




if __name__ == '__main__':
    
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    train=train_test.loc[train_test.set=='train','Hurricane_ID']
    generator=sample_generator(storm_list=train)
    

 
    generator.run()
  

    

